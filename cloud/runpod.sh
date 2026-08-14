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
#   RUNPOD_ROOT_SHA     encoded-root embedding sha the campaign must match (passed as
#                       TWM_ROOT_SHA). Defaults to the published root's sha; preflight
#                       fails closed on a mismatch. Re-pin after a deliberate re-encode.
#   RUNPOD_CONCURRENCY  jobs in flight per pod (passed as TWM_CONCURRENCY). Unset uses the
#                       runner's 3-per-GPU default; set 1 or 2 for a memory-heavy genome,
#                       which is what CUDA OOM on a shared card looks like. NOTE it also
#                       divides the CPU quota (runner.py: threads = cpu_quota() // conc),
#                       so LOWERING it RAISES BLAS threads per job (3->1 triples them). Aggregate
#                       demand stays inside the quota either way — only a value ABOVE the quota
#                       oversubscribes — but per-job thread count is what changes the score, so setting
#                       this appends '-c<N>' to ENV_TAG: every record then carries the value and
#                       `evolve doctor` flags the split. That is an audit trail, not a firewall —
#                       selection ignores env unless fitness.selection_env is pinned.
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
CONCURRENCY="${RUNPOD_CONCURRENCY:-}"
# The encoded root is pulled from HuggingFace rather than re-encoded per box, so its tensors are a
# fact about the published artefact, not about this pod. Pinning the sha makes eval/preflight
# REFUSE a root whose embeddings differ instead of scoring against them silently — the failure it
# catches is a wrong or stale root, which every other check passes happily. Override to re-pin
# after a deliberate re-encode + publish (the sha is printed by `pack_lane.sh prepare`).
ROOT_SHA="${RUNPOD_ROOT_SHA:-37801f89e25cdd3f3a8154d1fcd3b85c39c3ba8112b9d867e9588901f088d800}"
# Concurrency divides the CPU quota into per-job BLAS threads (runner.py), and thread count
# changes float reduction order, so it changes the score. Tagging puts that fact on every record
# and makes `evolve doctor` flag the split. It is NOT a firewall: this project has
# fitness.selection_env unset, and the archive filters by env only when that is pinned — so such
# runs still enter parent sampling. Pin selection_env if you need them excluded. Unset changes nothing.
[ -n "$CONCURRENCY" ] && ENV_TAG="$ENV_TAG-c$CONCURRENCY"
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
# The keepalive is what fails a dead peer, and the comparison that matters is against the flags
# these replace, not against a bare config. `ssh -G` on this host: with the OLD flags
# (-o ServerAliveInterval=30 alone) CountMax was inherited from ~/.ssh/config as 6, giving
# 30x6 = 180s; with these, 30x3 = 90s. So pinning CountMax halves dead-peer detection here, and
# makes it independent of whatever config the next operator has. ConnectTimeout is the second new
# bound: `ssh -G` resolves `connecttimeout none` without it, so an unanswered TCP handshake waited
# on the OS default.
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ConnectTimeout=30${RUNPOD_SSH_KEY:+ -i $RUNPOD_SSH_KEY}${RUNPOD_SSH_PIN:+ -o IdentitiesOnly=yes}"

# Every API call is bounded. terminate is a single gql() call, so an unbounded curl would wedge
# the one command whose entire purpose is to stop the meter, and lane.sh's terminate-confirmation
# loop shares it — a hang there means the `die` that is supposed to catch a failed stop never runs.
#
# Retry is deliberately NOT part of this: a deploy that times out at --max-time may already have
# been executed server-side, and retrying it would bill a second pod nobody is tracking. Retries
# are added per call, and only where the request is idempotent (a read).
CURL_TIMEOUTS="--connect-timeout 15 --max-time 60"
# Reads get a SHORTER per-attempt budget than the sequence cap, deliberately. --retry-max-time is
# only tested between attempts, so a cap below --max-time silently turns --retry 3 into --retry 0
# for a stalled-but-connected peer — the exact case retry exists for (measured: 1 attempt at
# max-time 60 / cap 45; 8 attempts once the cap sits above it). 25s is far above a healthy API
# response, and 4 attempts still fit inside the 120s sequence cap.
CURL_READ="--connect-timeout 15 --max-time 25 --retry 3 --retry-connrefused --retry-max-time 120"
# A deploy creates a billing resource. If curl gives up while the server is still working, the
# response carrying the pod id is lost but the pod may exist — and the ledger append below only
# runs once an id has been parsed, so that pod is never recorded and `reap` shows it as UNTRACKED
# to whoever thinks to look. Waiting for a slow deploy costs a minute; abandoning one costs a pod.
CURL_DEPLOY="--connect-timeout 15"

# GNU rsync aborts on --timeout (exit 30); openrsync does not, and there the flag can only turn a
# recovered stall into a failed transfer. GNU prints "rsync  version 3.x"; openrsync prints
# "openrsync: protocol version 29".
RSYNC_TIMEOUT=""
rsync --version 2>/dev/null | head -1 | grep -q '^rsync ' && RSYNC_TIMEOUT="--timeout=120"

log() { echo "==> $*" >&2; }
die() { echo "ERROR: $*" >&2; exit 1; }

# A retried read must not stream to stdout: curl re-sends the request but does NOT unwrite the
# bytes it already emitted, so a peer that sends half a body and stalls leaves the partial prefix
# concatenated ahead of the good body — at exit 0, for the caller's jq to choke on. Writing to a
# file makes curl truncate per attempt, so the caller sees one whole body or a non-zero status.
_curl_read() {
    local out rc=0
    out=$(mktemp) || return 1
    curl -sS $CURL_READ -o "$out" "$@" || rc=$?
    [ "$rc" = 0 ] && cat "$out"
    rm -f "$out"
    return "$rc"
}

gql() {
    local q; q=$(jq -Rn --arg q "$1" '{query:$q}')
    # A GraphQL query is a safe read and worth retrying through a flaky link; a mutation is not,
    # because a timed-out request may already have taken effect on the server.
    case "$1" in
        *mutation*) curl -sS $CURL_TIMEOUTS -X POST "$GQL" -H "Content-Type: application/json" -d "$q" ;;
        *)          _curl_read -X POST "$GQL" -H "Content-Type: application/json" -d "$q" ;;
    esac
}

# REST (Bearer auth). The two APIs are not interchangeable, so the split is deliberate: queries,
# status and terminate go over GraphQL, while every deploy goes over REST — network volumes and
# allowedCudaVersions exist only there, and the GraphQL podFindAndDeployOnDemand mutation has
# neither field.
REST_BASE="https://rest.runpod.io/v1"
rest() {
    local method="$1" path="$2" body="${3:-}"
    # Retry a GET; never a deploy, because a retried POST is a second pod. And a deploy gets no
    # --max-time at all: losing the response to a pod that was created is the expensive failure.
    if [ "$method" = "GET" ]; then
        _curl_read -X GET "$REST_BASE$path" -H "Authorization: Bearer $RUNPOD_API_KEY"
        return
    fi
    local flags="$CURL_TIMEOUTS"
    [ "$method" = "POST" ] && flags="$CURL_DEPLOY"
    if [ -n "$body" ]; then
        curl -sS $flags -X "$method" "$REST_BASE$path" -H "Authorization: Bearer $RUNPOD_API_KEY" \
            -H "Content-Type: application/json" -d "$body"
    else
        curl -sS $flags -X "$method" "$REST_BASE$path" -H "Authorization: Bearer $RUNPOD_API_KEY"
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
    # Bounded by WALL CLOCK, not iterations. Each pass makes two retried reads, and a retried read
    # against a stalled-but-connected API costs ~107s rather than failing immediately — so a
    # 60-iteration counter silently became hours, on a pod that is already deployed and billing
    # with nobody holding its id. The counter stays as a secondary guard.
    local ip="" port="" st="" tries=0 t0=$SECONDS budget=900
    while :; do
        [ $((SECONDS - t0)) -lt "$budget" ] || die "pod never exposed a public SSH port within ${budget}s (check the RunPod console for $id, and terminate it there if it is billing)"
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
    # Same rule as the no-volume path: a deploy that returned has already created a billing
    # resource, so record it before anything else can fail. Without this every volume deploy
    # is invisible to `reap`, which greps live pods against this file.
    _ledger "$id"
    echo "$id"
}

# No-volume deploy over REST so the request can constrain the host's CUDA driver version. The
# consumer-GPU fleet is heterogeneous and an old-driver host cannot initialise a modern torch
# wheel at all. allowedCudaVersions only biases placement, so the launch loop below still VERIFIES
# with nvidia-smi and re-rolls if the field was ignored for that host class.
ALLOWED_CUDA="${RUNPOD_ALLOWED_CUDA:-12.6,12.7,12.8,12.9,13.0}"
# Deploy ledger: every pod id this machine has created, appended the moment the API returns it.
LEDGER="${RUNPOD_LEDGER:-$HOME/.runpod-pods.log}"
_ledger() { mkdir -p "$(dirname "$LEDGER")" && echo "$(date -u +%FT%TZ) $1" >> "$LEDGER"; }
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
    _ledger "$id"
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
    _ssh_to "$ip" "$port" 'timeout 600 bash -c "(command -v rsync >/dev/null && command -v git >/dev/null && command -v tmux >/dev/null) || (apt-get update -qq && apt-get install -y -qq rsync git curl tmux)"'

    # Code only. data/ and enc/ are pulled on the box from HuggingFace, .results/ and
    # cloud/podresults/ are the box's own output, and .venv/ is built there against its driver.
    # --delete removes stale code, but rsync never deletes what an --exclude covers, so
    # re-bootstrapping to ship a code change leaves the encoded roots and finished results intact.
    log "rsync repo code → root@$ip:$POD_REPO_LIT/  (code only)"
    # What bounds a stalled transfer is the ssh keepalive above killing the transport child.
    # --timeout is NOT that bound and is not harmless: on openrsync (the macOS /usr/bin/rsync) it
    # logs 'poll: timeout' and keeps waiting on a true hang, while FAILING a stall that would have
    # recovered — measured, an 8s stall then normal service gives exit 1 and an empty destination
    # with the flag, and a completed transfer without it. It only delivers the abort on GNU rsync,
    # so it is applied only there, via $RSYNC_TIMEOUT.
    rsync -a --delete $RSYNC_TIMEOUT \
        --exclude=.venv --exclude=.git --exclude=__pycache__ --exclude='*.pyc' \
        --exclude='data/' --exclude='enc/' --exclude='ckpt/' \
        --exclude='.results/' --exclude='.cache/' --exclude='evolve/.cache/' \
        --exclude='cloud/podresults/' --exclude='*.pt' --exclude='*.safetensors' \
        -e "ssh $SSH_OPTS -p $port" "$REPO_DIR/" "root@$ip:$POD_REPO_LIT/"

    # Bounded like the apt step above, and for the same reason: ssh keepalives fail a DEAD peer,
    # but a stalled wheel download keeps the session alive, so nothing on the client side breaks
    # this hang. It is also the expensive step (~12GB of locked wheels, plus the CUDA-12 swap), so
    # an unbounded stall here is what bills a box overnight. Exit 124 surfaces as a loud failure.
    log "uv sync + CUDA smoke on the pod"
    _ssh_to "$ip" "$port" "
        set -e
        # Downloaded to a file, not piped: curl cannot rewind a pipe, so a retried download
        # feeds the truncated prefix to the shell once per attempt and the pipeline still exits 0.
        if ! command -v uv >/dev/null 2>&1; then
            # An || group exempts its non-final commands from set -e, so a failed download used to
            # sail past and surface later as a misleading 'uv: command not found'.
            # No --remove-on-error here: it needs curl >= 7.83 and this image is ubuntu 22.04
            # (curl 7.81), where an unknown option makes curl exit 2 and bootstrap fail on every
            # pod. The rm in the failure branch does the same job on any version.
            curl -LsSf --connect-timeout 15 --max-time 300 --retry 3 --retry-max-time 600 \
                 -o /tmp/uv-install.sh https://astral.sh/uv/install.sh \
                || { rm -f /tmp/uv-install.sh; echo 'uv installer download failed' >&2; exit 1; }
            sh /tmp/uv-install.sh
        fi
        export PATH=\"\$HOME/.local/bin:\$PATH\"
        # Bounded because a stalled wheel download keeps the ssh session alive, so no client-side
        # keepalive can break it — and this is the ~12GB step that bills a box overnight.
        cd $POD_REPO && timeout 2400 uv sync
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
            12.6*|12.7*|12.8*|12.9*) timeout 1200 uv pip install --index-url https://download.pytorch.org/whl/cu126 'torch==2.13.0+cu126' ;;
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
    # Resume, not safety: an interrupted transfer already leaves nothing at the destination name.
    # These are multi-GB packs over a home uplink, so re-sending from zero is the cost worth
    # avoiding; the dotted dir keeps the retained partial obvious and out of the data root proper.
    rsync -a $RSYNC_TIMEOUT --partial-dir=.rsync-partial -e "ssh $SSH_OPTS -p $port" "${src%/}/" "root@$ip:$POD_REPO_LIT/data/$dest/"
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
${CONCURRENCY:+export TWM_CONCURRENCY='$CONCURRENCY'}
${ROOT_SHA:+export TWM_ROOT_SHA='$ROOT_SHA'}   # preflight refuses a root whose embeddings differ   # jobs in flight per pod; unset means the runner's 3-per-GPU default
${TWM_RUN_ID:+export TWM_RUN_ID='$TWM_RUN_ID'}   # stamped into .done so a poller can tell THIS campaign's marker from a previous one's
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
    # rsync already writes to a temp name and discards it on interruption, so nothing short ever
    # appears at the destination name. --partial-dir is here to let an interrupted pull RESUME
    # instead of re-sending, and the dotted directory keeps the retained bytes out of the way of
    # anything that globs this directory (lane.sh's verify counts *.json here).
    rsync -a $RSYNC_TIMEOUT --partial-dir=.rsync-partial -e "ssh $SSH_OPTS -p $port" "root@$ip:$POD_REPO_LIT/cloud/podresults/" "$dest/"
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
