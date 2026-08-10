#!/usr/bin/env bash
# cloud/runpod.sh — rent a GPU box on RunPod, put this repo on it, run a measurement campaign,
# bring the results home. Pure curl + jq + ssh + rsync; no vendor CLI to install.
#
# A pod is a Docker container. You get `root` over SSH on an EXPOSED PUBLIC PORT (not user@ip:22),
# and the ssh public key is injected through the pod's PUBLIC_KEY environment variable.
#
# Subcommands (each prints machine-parseable results on the LAST stdout line):
#   types [--available]           list GPU types with price + stock (filter: $RUNPOD_TYPES_FILTER)
#   launch                        deploy a pod; prints "<podId> <ip> <port>"
#   host      <podId>             re-resolve "<ip> <port>" (survives a pod restart)
#   ssh       <podId>             interactive ssh into the pod
#   bootstrap <podId>             apt deps + rsync this repo's code + uv sync + CUDA smoke test
#   campaign  <podId> [genome.json...]
#                                 run cloud/pack_lane.sh prepare, then pack_lane.sh campaign over
#                                 the genomes (default: all of evolve/genomes/*.json), inside tmux
#   push-raw  <podId> <localDir> <destSubdir>
#                                 rsync a data directory that is NOT published to HuggingFace
#                                 (e.g. the raw cd-history records pack_lane.sh wants in
#                                 TWM_CDH_RAW). Published roots are pulled by the box itself.
#   pull      <podId>             rsync the pod's cloud/podresults/ down into this repo
#   terminate <podId>             TERMINATE THE POD — this is what stops billing
#   status    <podId> | list
#   reap                          list every pod on the account against the ids this machine
#                                 recorded at deploy time, so a launch that was interrupted before
#                                 it could report an id still shows up instead of billing unnoticed
#   datacenters | volumes | create-volume <name> <gb> <dc>    (network-volume path; unused by
#                                 default — the box pulls its data roots from HuggingFace)
#
# The data roots and the encoder are NOT copied up from here. cloud/pack_lane.sh downloads them on
# the box from HuggingFace at datacenter bandwidth, which beats a home uplink by a wide margin.
# Only code goes over the wire.
#
# Auth: RUNPOD_API_KEY, sourced from $RUNPOD_ENV_FILE (default ~/.runpod.env). The key is never
# echoed or logged. `pack_lane.sh publish` additionally needs a HuggingFace write token exported on
# the box; set it there by hand, it is not forwarded from here.
#
# Env knobs:
#   RUNPOD_GPU_TYPE     default "NVIDIA GeForce RTX 4090"
#   RUNPOD_GPU_COUNT    default 1        (also becomes TWM_GPUS for the campaign)
#   RUNPOD_CLOUD        SECURE | COMMUNITY | ALL   (default SECURE; COMMUNITY is cheaper)
#   RUNPOD_IMAGE        default runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#                       (uv sync installs this repo's own python + torch; the image only has to
#                       supply a working CUDA userspace)
#   RUNPOD_DISK_GB      default 100 — see the sizing comment beside DISK_GB below
#   RUNPOD_POD_NAME     default twm-pack
#   RUNPOD_ENV_TAG      environment tag the scores are ingested under; default derives from the GPU
#                       type, e.g. runpod-4090. Scores only compare within one tag.
#   RUNPOD_SEEDS        seeds per genome (passed as TWM_SEEDS; pack_lane's own default is 0,1,2)
#   RUNPOD_TYPES_FILTER default "4090|5090|A100|H100|L40|A40|RTX 6000"
#   RUNPOD_ALLOWED_CUDA host driver CUDA versions to accept; default 12.6,12.7,12.8,12.9,13.0
#   RUNPOD_PUBKEY_FILE / RUNPOD_SSH_KEY   ssh identity (auto-discovered otherwise)
#   RUNPOD_USE_VOLUME=1 + RUNPOD_VOLUME_ID + RUNPOD_DATACENTER (+ RUNPOD_VOLUME_MOUNT)
#                       attaching a network volume is an explicit opt-in, because a shared env file
#                       can carry a stale volume id that silently pins the deploy to a dead
#                       datacenter
#
# Typical run:
#   read -r POD IP PORT < <(cloud/runpod.sh launch | tail -1)
#   cloud/runpod.sh bootstrap "$POD"
#   cloud/runpod.sh campaign  "$POD"        # detaches into tmux; prints how to follow it
#   cloud/runpod.sh pull      "$POD"
#   cloud/runpod.sh terminate "$POD"        # an idle box bills by the second

set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the repo root; cloud/ sits in it
POD_REPO='$HOME/terminal-wm'                                  # where the code lives on the box
POD_REPO_LIT="~/terminal-wm"                                  # same path, for rsync/messages

RUNPOD_ENV_FILE="${RUNPOD_ENV_FILE:-$HOME/.runpod.env}"
[ -f "$RUNPOD_ENV_FILE" ] && { set +u; . "$RUNPOD_ENV_FILE"; set -u; }
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY required (put it in $RUNPOD_ENV_FILE)}"

GQL="https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY"
GPU_TYPE="${RUNPOD_GPU_TYPE:-NVIDIA GeForce RTX 4090}"
GPU_COUNT="${RUNPOD_GPU_COUNT:-1}"
CLOUD="${RUNPOD_CLOUD:-SECURE}"
IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
# Disk sizing, in round numbers: the uv venv with CUDA torch is ~12GB and the driver-matched swap
# below can leave a second set of CUDA wheels resident (~10GB more); the uv wheel cache holds
# roughly one more copy; the encoder checkpoint plus the raw and encoded pack roots pulled from
# HuggingFace are a few GB; and the campaign writes a shared lane context plus per-(genome,seed)
# scratch under .results/, which grows with the size of the genome set. 100GB leaves headroom for
# all of that. Container disk is cheap; running out mid-campaign is not.
DISK_GB="${RUNPOD_DISK_GB:-100}"
POD_NAME="${RUNPOD_POD_NAME:-twm-pack}"
TYPES_FILTER="${RUNPOD_TYPES_FILTER:-4090|5090|A100|H100|L40|A40|RTX 6000}"
# The tag every score from this box is ingested under. Scores compare only within one environment,
# so the tag names the hardware: "NVIDIA GeForce RTX 4090" -> runpod-4090.
ENV_TAG="${RUNPOD_ENV_TAG:-runpod-$(printf '%s' "$GPU_TYPE" | awk '{print tolower($NF)}')}"
SEEDS="${RUNPOD_SEEDS:-}"
TMUX_SESSION="${RUNPOD_TMUX_SESSION:-twm-campaign}"

RUNPOD_SSH_KEY="${RUNPOD_SSH_KEY:-}"
if [ -z "$RUNPOD_SSH_KEY" ]; then
    for c in "$HOME/.ssh/id_ed25519" "$HOME/.ssh/id_rsa"; do
        [ -f "$c" ] && RUNPOD_SSH_KEY="$c" && break
    done
fi
# accept-new: every pod is a fresh host key, so strict checking only produces prompts. The chosen
# key is OFFERED with -i but not hard-pinned: RunPod also honours the account-level console key in
# authorized_keys, and a pod can end up accepting that one instead of the injected PUBLIC_KEY —
# IdentitiesOnly=yes would then lock you out of your own box. Set RUNPOD_SSH_PIN=1 to pin anyway,
# which is what you want if a crowded ssh-agent trips the server's MaxAuthTries.
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30${RUNPOD_SSH_KEY:+ -i $RUNPOD_SSH_KEY}${RUNPOD_SSH_PIN:+ -o IdentitiesOnly=yes}"

log() { echo "==> $*" >&2; }
die() { echo "ERROR: $*" >&2; exit 1; }

gql() {
    local q; q=$(jq -Rn --arg q "$1" '{query:$q}')
    curl -sS -X POST "$GQL" -H "Content-Type: application/json" -d "$q"
}

# REST (Bearer auth). The two APIs are not interchangeable, so the split is deliberate: queries,
# status and terminate go over GraphQL, while every deploy goes over REST — network volumes and
# allowedCudaVersions exist only there, and the GraphQL podFindAndDeployOnDemand mutation has
# neither field.
REST_BASE="https://rest.runpod.io/v1"
rest() {
    local method="$1" path="$2" body="${3:-}"
    if [ -n "$body" ]; then
        curl -sS -X "$method" "$REST_BASE$path" -H "Authorization: Bearer $RUNPOD_API_KEY" \
            -H "Content-Type: application/json" -d "$body"
    else
        curl -sS -X "$method" "$REST_BASE$path" -H "Authorization: Bearer $RUNPOD_API_KEY"
    fi
}

_pubkey() {
    local f="${RUNPOD_PUBKEY_FILE:-${RUNPOD_SSH_KEY:+$RUNPOD_SSH_KEY.pub}}"
    [ -n "$f" ] && [ -f "$f" ] || die "no ssh public key (set RUNPOD_PUBKEY_FILE, or RUNPOD_SSH_KEY with a .pub beside it)"
    cat "$f"
}

# resolve a running pod's public ssh endpoint as "<ip> <port>" (the privatePort-22 mapping)
_host() {
    local id="$1" js
    js=$(gql "query{ pod(input:{podId:\"$id\"}){ id desiredStatus runtime{ ports{ ip isIpPublic privatePort publicPort type } } } }")
    echo "$js" | jq -r '.data.pod.runtime.ports[]? | select(.privatePort==22 and .isIpPublic==true) | "\(.ip) \(.publicPort)"' | head -1
}

_ssh_to() { local ip="$1" port="$2"; shift 2; ssh $SSH_OPTS -p "$port" "root@$ip" "$@"; }

# resolve <podId> into the caller's $ip/$port, or die
_need_host() {
    local id="$1"
    read -r ip port < <(_host "$id") || true
    [ -n "${ip:-}" ] && [ -n "${port:-}" ] || die "no public SSH port for $id (is it running? try: $0 status $id)"
}

cmd_types() {
    local sel=""
    [ "${1:-}" = "--available" ] && sel='| select(.lowestPrice.stockStatus != null)'
    gql "query{ gpuTypes{ id displayName secureCloud communityCloud lowestPrice(input:{gpuCount:$GPU_COUNT}){ uninterruptablePrice stockStatus } } }" \
        | jq -r ".data.gpuTypes[]? | select(.displayName|test(\"$TYPES_FILTER\")) $sel | \"\(.displayName) | ${GPU_COUNT}x stock:\(.lowestPrice.stockStatus // \"none\") \$\(.lowestPrice.uninterruptablePrice // \"-\")/hr | id=\(.id)\""
}

# poll a just-created pod id until it is RUNNING with a public SSH port, wait for sshd, then print
# "<id> <ip> <port>" as the parseable last line.
_await_ssh() {
    local id="$1"
    log "pod id: $id — polling for RUNNING + a public SSH port (2-5 min)"
    local ip="" port="" st="" tries=0
    while :; do
        st=$(gql "query{ pod(input:{podId:\"$id\"}){ desiredStatus } }" 2>/dev/null | jq -r '.data.pod.desiredStatus // "?"' 2>/dev/null || echo "?")
        # Reset the vars and swallow the read's exit status: before the port exists _host emits
        # nothing, `read` hits EOF and returns non-zero, and under `set -e` that would abort the
        # launch while leaving a pod deployed and BILLING with nobody holding its id.
        ip=""; port=""; read -r ip port < <(_host "$id") || true
        echo "    status=$st ssh=${ip:+$ip:$port}" >&2
        [ -n "${ip:-}" ] && [ -n "${port:-}" ] && break
        tries=$((tries+1)); [ "$tries" -gt 60 ] && die "pod never exposed a public SSH port (check the RunPod console for $id, and terminate it there if it is billing)"
        sleep 10
    done
    log "waiting for sshd at $ip:$port"
    for _ in $(seq 1 30); do _ssh_to "$ip" "$port" true 2>/dev/null && break; sleep 10; done
    echo "$id $ip $port"     # last line: parseable
}

_deploy_rest() {
    local pubkey; pubkey=$(_pubkey | tr -d '\n')
    : "${RUNPOD_DATACENTER:?RUNPOD_DATACENTER required with RUNPOD_VOLUME_ID}"
    local mount="${RUNPOD_VOLUME_MOUNT:-/workspace}"
    log "deploying ${GPU_COUNT}x '$GPU_TYPE' ($CLOUD) in $RUNPOD_DATACENTER, volume $RUNPOD_VOLUME_ID @ $mount"
    local body r id
    body=$(jq -n --arg name "$POD_NAME" --arg img "$IMAGE" --arg cloud "$CLOUD" \
        --argjson gc "$GPU_COUNT" --arg gt "$GPU_TYPE" --arg dc "$RUNPOD_DATACENTER" \
        --arg vol "$RUNPOD_VOLUME_ID" --arg mount "$mount" --argjson disk "$DISK_GB" --arg pk "$pubkey" \
        '{name:$name, imageName:$img, cloudType:$cloud, computeType:"GPU",
          gpuCount:$gc, gpuTypeIds:[$gt], dataCenterIds:[$dc],
          networkVolumeId:$vol, volumeMountPath:$mount, volumeInGb:0,
          containerDiskInGb:$disk, ports:["22/tcp"], env:{PUBLIC_KEY:$pk}}')
    r=$(rest POST /pods "$body")
    id=$(echo "$r" | jq -r '.id // empty')
    [ -n "$id" ] || { echo "$r" | jq -r '.error // .message // .' >&2; die "deploy (REST + volume) failed"; }
    echo "$id"
}

# No-volume deploy over REST so the request can constrain the host's CUDA driver version. The
# consumer-GPU fleet is heterogeneous and an old-driver host cannot initialise a modern torch
# wheel at all. allowedCudaVersions only biases placement, so the launch loop below still VERIFIES
# with nvidia-smi and re-rolls if the field was ignored for that host class.
ALLOWED_CUDA="${RUNPOD_ALLOWED_CUDA:-12.6,12.7,12.8,12.9,13.0}"
# Deploy ledger: every pod id this machine has created, appended the moment the API returns it.
LEDGER="${RUNPOD_LEDGER:-$HOME/.runpod-pods.log}"
_deploy_rest_novol() {
    local pubkey; pubkey=$(_pubkey | tr -d '\n')
    log "deploying ${GPU_COUNT}x '$GPU_TYPE' ($CLOUD), disk=${DISK_GB}GB, allowedCuda=[$ALLOWED_CUDA]"
    local body r id
    body=$(jq -n --arg name "$POD_NAME" --arg img "$IMAGE" --arg cloud "$CLOUD" \
        --argjson gc "$GPU_COUNT" --arg gt "$GPU_TYPE" --argjson disk "$DISK_GB" \
        --arg pk "$pubkey" --arg cuda "$ALLOWED_CUDA" \
        '{name:$name, imageName:$img, cloudType:$cloud, computeType:"GPU",
          gpuCount:$gc, gpuTypeIds:[$gt], volumeInGb:0, containerDiskInGb:$disk,
          ports:["22/tcp"], env:{PUBLIC_KEY:$pk},
          allowedCudaVersions:($cuda | split(","))}')
    r=$(rest POST /pods "$body")
    id=$(echo "$r" | jq -r '.id // empty')
    [ -n "$id" ] || { echo "$r" | jq -r '.error // .message // .' >&2; die "deploy (REST no-volume) failed"; }
    # Record the id the instant it exists, BEFORE anything else can fail or be interrupted. A
    # deploy that returns has already created a billing resource; if the caller dies between here
    # and printing the id, the only trace left is this file. `reap` reads it.
    mkdir -p "$(dirname "$LEDGER")" && echo "$(date -u +%FT%TZ) $id" >> "$LEDGER"
    echo "$id"
}

_driver_ok() {  # <ip> <port> — true iff the host driver's CUDA version is >= 12.6
    local drv
    drv=$(_ssh_to "$1" "$2" 'nvidia-smi | grep -oE "CUDA Version: [0-9]+\.[0-9]+"' 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' || echo 0)
    log "host driver CUDA: $drv"
    awk -v d="$drv" 'BEGIN{exit !(d >= 12.6)}'
}

cmd_launch() {
    local id ip port try
    # A volume deploy is an explicit opt-in. A shared env file can carry a volume id from some
    # other project; pinning to a dead volume's datacenter fails the deploy with a message that
    # looks nothing like the real cause.
    if [ "${RUNPOD_USE_VOLUME:-0}" = "1" ] && [ -n "${RUNPOD_VOLUME_ID:-}" ]; then
        id=$(_deploy_rest); _await_ssh "$id"; return
    fi
    for try in 1 2 3 4; do
        id=$(_deploy_rest_novol)
        ip=""; port=""; read -r id ip port < <(_await_ssh "$id" | tail -1) || true
        [ -n "${ip:-}" ] || die "launch: no ssh endpoint for $id"
        if _driver_ok "$ip" "$port"; then echo "$id $ip $port"; return; fi
        log "driver too old on $id (try $try/4) — terminating and re-rolling"
        cmd_terminate "$id"
        sleep 5
    done
    die "no pod with driver CUDA >= 12.6 after 4 tries (widen RUNPOD_ALLOWED_CUDA or pick another GPU type)"
}

cmd_datacenters() {
    log "storage-capable datacenters + ${GPU_COUNT}x '$GPU_TYPE' stock:"
    local dc st
    for dc in $(gql 'query{ dataCenters { id storageSupport } }' | jq -r '.data.dataCenters[]? | select(.storageSupport==true) | .id'); do
        st=$(gql "query{ gpuTypes(input:{id:\"$GPU_TYPE\"}){ lowestPrice(input:{gpuCount:$GPU_COUNT, dataCenterId:\"$dc\"}){ stockStatus uninterruptablePrice } } }" \
            | jq -r '.data.gpuTypes[0].lowestPrice | "stock=\(.stockStatus // "none") $\(.uninterruptablePrice // "-")/hr"' 2>/dev/null)
        printf '  %-10s %s\n' "$dc" "$st" >&2
    done
}

cmd_volumes() {
    rest GET /networkvolumes | jq -r 'if type=="array" then (.[] | "id=\(.id)  name=\(.name)  size=\(.size)GB  dc=\(.dataCenterId)") elif .error then "ERROR: \(.error)" else . end'
}

cmd_create_volume() {
    local name="${1:?create-volume <name> <sizeGB> <dataCenterId>}" size="${2:?<sizeGB>}" dc="${3:?<dataCenterId>}"
    local body r id
    body=$(jq -n --arg n "$name" --argjson s "$size" --arg dc "$dc" '{name:$n, size:$s, dataCenterId:$dc}')
    r=$(rest POST /networkvolumes "$body")
    id=$(echo "$r" | jq -r '.id // empty')
    [ -n "$id" ] || { echo "$r" | jq -r '.error // .message // .' >&2; die "volume create failed"; }
    echo "$id"
}

cmd_host()   { local id="${1:?host <podId>}"; _host "$id"; }
cmd_status() { local id="${1:?status <podId>}"; gql "query{ pod(input:{podId:\"$id\"}){ id name desiredStatus machineId } }" | jq '.data.pod'; }
cmd_list()   { gql 'query{ myself{ pods{ id name desiredStatus machine{ gpuDisplayName } } } }' | jq '.data.myself.pods'; }

cmd_ssh() {
    local id="${1:?ssh <podId>}" ip port
    _need_host "$id"
    log "ssh root@$ip -p $port"
    exec ssh $SSH_OPTS -p "$port" "root@$ip"
}

cmd_bootstrap() {
    local id="${1:?bootstrap <podId>}" ip port
    _need_host "$id"

    # The base image may have none of these, and the first rsync needs rsync on both ends.
    log "ensuring rsync/git/curl/tmux on the pod"
    _ssh_to "$ip" "$port" '(command -v rsync >/dev/null && command -v git >/dev/null && command -v tmux >/dev/null) || (apt-get update -qq && apt-get install -y -qq rsync git curl tmux)'

    # Code only. data/ and enc/ are pulled on the box from HuggingFace, .results/ and
    # cloud/podresults/ are the box's own output, and .venv/ is built there against its driver.
    # --delete removes stale code, but rsync never deletes what an --exclude covers, so
    # re-bootstrapping to ship a code change leaves the encoded roots and finished results intact.
    log "rsync repo code → root@$ip:$POD_REPO_LIT/  (code only)"
    rsync -a --delete \
        --exclude=.venv --exclude=.git --exclude=__pycache__ --exclude='*.pyc' \
        --exclude='data/' --exclude='enc/' --exclude='ckpt/' \
        --exclude='.results/' --exclude='.cache/' --exclude='evolve/.cache/' \
        --exclude='cloud/podresults/' --exclude='*.pt' --exclude='*.safetensors' \
        -e "ssh $SSH_OPTS -p $port" "$REPO_DIR/" "root@$ip:$POD_REPO_LIT/"

    log "uv sync + CUDA smoke on the pod"
    _ssh_to "$ip" "$port" "
        set -e
        command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH=\"\$HOME/.local/bin:\$PATH\"
        cd $POD_REPO && uv sync
        # The lockfile pins the default CUDA-13 torch wheels, and much of the rented fleet still
        # runs a 12.x driver, where those binaries cannot initialise. Detect the driver's CUDA
        # version and swap in the matching CUDA-12 torch INSIDE THE POD VENV ONLY — the lockfile is
        # never touched, and UV_NO_SYNC=1 on later \`uv run\` calls stops uv restoring the locked
        # wheels underneath the swap.
        drv=\$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+')
        # The +cu126 local-version suffix is required: a bare torch spec counts as already
        # satisfied by the locked wheels and uv skips the swap entirely. cu126 is the newest
        # CUDA-12 index that still ships torch 2.13.0, and its wheels run on any >=12.6 driver, so
        # the pod ends up on exactly the torch version the lockfile names.
        case \"\$drv\" in
            12.6*|12.7*|12.8*|12.9*) uv pip install --index-url https://download.pytorch.org/whl/cu126 'torch==2.13.0+cu126' ;;
            12.*) echo \"driver CUDA \$drv is too old for a CUDA-12 torch 2.13 wheel — launch again for a newer-driver host\" >&2; exit 1 ;;
            *)    echo \"driver CUDA \$drv — keeping the locked CUDA-13 torch wheels\" ;;
        esac
        UV_NO_SYNC=1 uv run python -c \"import torch; assert torch.cuda.is_available(), 'no CUDA'; print('cuda ok:', torch.cuda.get_device_name(0), '| driver CUDA \$drv')\"
    "
    log "bootstrap complete for $id"
    echo "$id ready"
}

# push-raw <podId> <localDir> <destSubdir> — for data that is NOT in the published HuggingFace
# repo, which is the only kind worth pushing over a home uplink. Everything published is pulled by
# the box itself, far faster. The destination is under the pod's data root, so pack_lane.sh can be
# pointed at it (e.g. TWM_CDH_RAW=~/terminal-wm/data/<destSubdir>).
cmd_push_raw() {
    local id="${1:?push-raw <podId> <localDir> <destSubdir>}" src="${2:?<localDir>}" dest="${3:?<destSubdir>}" ip port
    _need_host "$id"
    [ -d "$src" ] || die "no such directory: $src"
    log "rsync $src → root@$ip:$POD_REPO_LIT/data/$dest/"
    _ssh_to "$ip" "$port" "mkdir -p $POD_REPO/data/$dest"
    rsync -a -e "ssh $SSH_OPTS -p $port" "${src%/}/" "root@$ip:$POD_REPO_LIT/data/$dest/"
    echo "$POD_REPO_LIT/data/$dest"
}

# campaign <podId> [genome.json...] — the whole point of renting the box. Runs pack_lane.sh
# prepare (pull + pin the encoder, pull and encode the pack root) and then pack_lane.sh campaign
# over the genomes, in that order, inside a detached tmux session. Pod ssh drops often enough that
# a multi-hour run tied to the connection will not survive; tmux keeps it alive on the box and lets
# you reattach after a drop.
cmd_campaign() {
    local id="${1:?campaign <podId> [genome.json...]}"; shift || true
    local ip port; _need_host "$id"

    _ssh_to "$ip" "$port" "test -x $POD_REPO/cloud/pack_lane.sh" \
        || die "$POD_REPO_LIT/cloud/pack_lane.sh is missing on $id — run: $0 bootstrap $id"
    ! _ssh_to "$ip" "$port" "tmux has-session -t $TMUX_SESSION" 2>/dev/null \
        || die "a '$TMUX_SESSION' session is already running on $id — follow it with: $0 ssh $id  then  tmux attach -t $TMUX_SESSION"

    # Genome arguments are repo-relative on the box. Accept a path from here in any shape and
    # reduce it to that: strip the local repo prefix, and fall back to the genome directory for a
    # bare name or an absolute path from somewhere else.
    local g rel genomes=""
    for g in "$@"; do
        rel="${g#"$REPO_DIR"/}"
        case "$rel" in
            /*)  rel="evolve/genomes/$(basename "$g")" ;;   # absolute, from outside this repo
            */*) : ;;                                       # already repo-relative
            *)   rel="evolve/genomes/$rel" ;;               # a bare genome name
        esac
        genomes="$genomes '$rel'"
    done
    # No genomes named: measure the whole starting population. The glob is left unquoted so the
    # POD's shell expands it against the code that is actually on the box.
    [ -n "$genomes" ] || genomes="evolve/genomes/*.json"

    local logfile="cloud/podresults/campaign.log"
    log "launching campaign on $id in tmux session '$TMUX_SESSION' (env tag: $ENV_TAG)"

    # Ship the run as a script rather than a giant one-liner: one level of quoting instead of
    # three, and it stays on the box as a record of exactly what was run.
    _ssh_to "$ip" "$port" "cat > $POD_REPO/twm_campaign.sh" <<EOF
#!/usr/bin/env bash
# Written by cloud/runpod.sh campaign. Runs under tmux on this box.
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
export UV_NO_SYNC=1                 # keep the driver-matched torch bootstrap installed
export TWM_ENV_TAG='$ENV_TAG'       # scores compare only within one environment tag
export TWM_GPUS='$GPU_COUNT'
${SEEDS:+export TWM_SEEDS='$SEEDS'}
cd "$POD_REPO"
mkdir -p cloud/podresults
exec > >(tee -a "$logfile") 2>&1
echo "=== prepare  \$(date -u +%FT%TZ)"
./cloud/pack_lane.sh prepare
echo "=== campaign \$(date -u +%FT%TZ)"
./cloud/pack_lane.sh campaign $genomes
echo "=== done     \$(date -u +%FT%TZ)"
EOF
    # $POD_REPO is left unquoted in the remote command on purpose: the POD's shell has to be the one
    # that expands $HOME. tmux gets the session's start directory with -c, so the command it runs
    # can stay a plain relative path and needs no expansion of its own.
    _ssh_to "$ip" "$port" "cd $POD_REPO && chmod +x twm_campaign.sh && tmux new-session -d -s $TMUX_SESSION -c $POD_REPO 'bash twm_campaign.sh'"

    cat >&2 <<EOF

campaign running detached on $id.

  follow it:      $0 ssh $id     then:  tmux attach -t $TMUX_SESSION   (detach again with ctrl-b d)
  or tail the log: ssh $SSH_OPTS -p $port root@$ip 'tail -f $POD_REPO_LIT/$logfile'

when it finishes, fold each genome into an ingestable record on the box ($0 ssh $id):
  ./cloud/pack_lane.sh score evolve/genomes/<name>.json <candidate-id>
(the per-seed work is already cached, so that step only aggregates)

then bring everything home and STOP THE BILLING:
  $0 pull $id
  $0 terminate $id        <-- an idle box bills by the second

EOF
    echo "$id $TMUX_SESSION $logfile"     # last line: parseable
}

cmd_pull() {
    local id="${1:?pull <podId>}" ip port
    _need_host "$id"
    local dest="$REPO_DIR/cloud/podresults"; mkdir -p "$dest"
    log "rsync pod cloud/podresults/ → $dest/"
    rsync -a -e "ssh $SSH_OPTS -p $port" "root@$ip:$POD_REPO_LIT/cloud/podresults/" "$dest/"
    log "pulled. If the box has no more work to do: $0 terminate $id"
    echo "$dest"
}

cmd_reap() {
    # Every pod this account has, against the ids this machine recorded at deploy time. An
    # interrupted launch is the common way to end up with a running pod nobody is holding: the
    # deploy call succeeds, the caller dies before reporting, and the box bills quietly. Anything
    # listed as UNTRACKED or as tracked-but-forgotten is a candidate for termination — this only
    # ever REPORTS, so deciding what dies stays with a person.
    local live tracked
    live=$(gql 'query{ myself{ pods{ id name desiredStatus } } }' \
           | jq -r '.data.myself.pods[]? | "\(.id) \(.name) \(.desiredStatus)"')
    tracked=$( [ -f "$LEDGER" ] && awk '{print $2}' "$LEDGER" || true )
    [ -n "$live" ] || { log "no pods on this account"; return 0; }
    echo "$live" | while read -r id name st; do
        if echo "$tracked" | grep -qx "$id"; then echo "  tracked   $id  $name  $st"
        else echo "  UNTRACKED $id  $name  $st"; fi
    done
    echo
    log "terminate anything you do not recognise:  $0 terminate <podId>"
}

cmd_terminate() {
    local id="${1:?terminate <podId>}"
    log "terminating $id"
    gql "mutation{ podTerminate(input:{podId:\"$id\"}) }" | jq -r 'if .errors then .errors[0].message else "terminated" end' >&2
}

case "${1:-}" in
    ""|-h|--help)
        # the leading comment block is the help text
        awk 'NR>1 { if (!/^#/) exit; sub(/^# ?/, ""); print }' "$0" >&2
        exit 0 ;;
esac
CMD="$1"; shift
case "$CMD" in
    types) cmd_types "$@";; datacenters) cmd_datacenters "$@";;
    volumes) cmd_volumes "$@";; create-volume) cmd_create_volume "$@";;
    launch) cmd_launch "$@";; host) cmd_host "$@";;
    ssh) cmd_ssh "$@";; bootstrap) cmd_bootstrap "$@";;
    push-raw) cmd_push_raw "$@";; campaign) cmd_campaign "$@";;
    pull) cmd_pull "$@";;
    terminate) cmd_terminate "$@";; status) cmd_status "$@";; list) cmd_list "$@";;
    reap) cmd_reap "$@";;
    *) die "unknown command: $CMD";;
esac
