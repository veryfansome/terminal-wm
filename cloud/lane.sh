#!/usr/bin/env bash
# One command from nothing to ingestable records, with the pod verified stopped at the end.
#
#   cloud/lane.sh run <genome.json>...     provision -> bootstrap -> campaign -> poll ->
#                                          download -> verify -> terminate
#   STAGE=<name> cloud/lane.sh run ...     run a single stage against the recorded pod
#   POD=<id>     cloud/lane.sh run ...     attach to a pod instead of provisioning one
#   FORCE=1      cloud/lane.sh run ...     re-measure genomes that already have a local record
#   LANE_STATE=<path>                      where this lane records its pod id. Give two
#                                          concurrent lanes DIFFERENT paths and DISJOINT
#                                          genome sets to halve wall-clock on two 1x boxes.
#
# Two failures on this lane cost real money, and neither is one a person reliably avoids by
# intending to. A campaign finished and nothing noticed, so a box billed for hours doing nothing.
# And a pod was terminated before its per-seed metrics were folded and copied home, throwing away
# a measurement already paid for. Every guard below exists because of one of those, or because a
# review found the guard itself inert.
#
# The load-bearing rules:
#   - verify GATES terminate, and verify tests records on LOCAL disk, not a promise;
#   - a failure leaves the pod RUNNING and says so, because an idle box costs cents an hour
#     while a lost campaign costs the campaign;
#   - "unreachable" is never evidence of anything — not that the run finished, not that it died;
#   - the completion marker is attributed to THIS dispatch by nonce, because a marker left by an
#     earlier campaign is indistinguishable from instant success;
#   - termination is confirmed against the provider, because the terminate call exits 0 on an
#     API error.
set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
RP="bash cloud/runpod.sh"
# Per-lane, so two lanes can run side by side on disjoint genome sets without each
# overwriting the other's pod id — which would leave one box billing with nobody holding it.
STATE="${LANE_STATE:-cloud/.lane-pod}"
POD_REPO="~/terminal-wm"
TMUX_SESSION="twm-campaign"
POLL_MAX_MIN="${POLL_MAX_MIN:-360}"
POD="${POD:-}"
LANE_ARGS=""

log() { printf '[lane %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { printf '[lane] ERROR: %s\n' "$*" >&2; exit 1; }

# An EXIT trap, not ERR: ERR is not inherited into functions without -E and never fires for the
# explicit `exit` inside die(), which is how every guard here reports. The pod id is printed
# because terminate truncates the state file, and the genome list comes from a captured global
# because a trap fires in the frame being exited, where "$*" is whatever that frame's args were.
_on_exit() {
    [ "${1:-0}" = 0 ] && return 0
    local id; id="${POD:-$( [ -s "$STATE" ] && cat "$STATE" || true )}"
    [ -n "$id" ] || return 0
    printf '[lane] FAILED (rc=%s) — pod %s is STILL RUNNING and billing.\n' "$1" "$id" >&2
    printf '       Look first:     cloud/runpod.sh ssh %s   then: tmux attach -t %s\n' "$id" "$TMUX_SESSION" >&2
    printf '       Then finish:    STAGE=terminate cloud/lane.sh run %s\n' "$LANE_ARGS" >&2
    printf '       Or stop it now: cloud/runpod.sh terminate %s && cloud/runpod.sh reap\n' "$id" >&2
}

# runpod.sh only auto-discovers ~/.ssh/id_ed25519 and ~/.ssh/id_rsa, and fails AFTER deploying.
if [ -z "${RUNPOD_SSH_KEY:-}" ]; then
    for c in "$HOME/.ssh/id_ed25519" "$HOME/.ssh/id_rsa" "$HOME/.ssh/lambda_cloud_ed25519"; do
        [ -f "$c" ] && [ -s "$c" ] && [ -f "$c.pub" ] && { RUNPOD_SSH_KEY="$c"; break; }
    done
fi
[ -n "${RUNPOD_SSH_KEY:-}" ] || die "no ssh key with a matching non-empty .pub — set RUNPOD_SSH_KEY"
export RUNPOD_SSH_KEY

_pod() {
    [ -n "$POD" ] && { echo "$POD"; return; }
    [ -s "$STATE" ] || die "no pod recorded in $STATE — run provision, or pass POD=<id>"
    cat "$STATE"
}

_ssh() {   # returns 255 when the host cannot be reached, which callers MUST distinguish
    local h ip port
    h="$($RP host "$(_pod)" 2>/dev/null | tail -1)" || return 255
    [ -n "$h" ] || return 255
    ip="${h%% *}"; port="${h##* }"
    ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=25 \
        -i "$RUNPOD_SSH_KEY" -p "$port" "root@$ip" "$@"
}

_result_for() { echo "cloud/podresults/$(basename "$1" .json).json"; }

# ok = a real measurement; failed = a measured candidate failure, ingestable; bad = nothing
# usable. Testing for the string "fitness" would pass a {"fitness": null} failure record, which
# is how a gate becomes a tautology.
_record_state() {
    local f="$1"
    [ -s "$f" ] || { echo bad; return; }
    jq -e '.correct == true and (.fitness | type) == "number"' "$f" >/dev/null 2>&1 && { echo ok; return; }
    jq -e '.correct == false' "$f" >/dev/null 2>&1 && { echo failed; return; }
    echo bad
}

_pending() {
    local g f
    for g in "$@"; do
        f="$(_result_for "$g")"
        if [ "${FORCE:-0}" != "1" ] && [ "$(_record_state "$f")" = ok ]; then continue; fi
        echo "$g"
    done
}

stage_provision() {
    if [ -n "$POD" ] || [ -s "$STATE" ]; then log "provision: reusing $(_pod)"; return; fi
    log "provision: deploying"
    local out; out="$($RP launch | tail -1)"
    echo "${out%% *}" > "$STATE"
    log "provision: $(cat "$STATE")"
}

stage_bootstrap() { log "bootstrap: $(_pod)"; $RP bootstrap "$(_pod)"; }

stage_campaign() {
    local todo; todo="$(_pending "$@")"
    [ -n "$todo" ] || { log "campaign: nothing pending"; return; }
    # Synchronous with dispatch: clearing it inside campaign() on the box is too late, because
    # `prepare` runs first and can take minutes while the poller is already looking.
    _ssh "rm -f $POD_REPO/cloud/podresults/.done" \
        || die "cannot reach the pod to clear the previous completion marker"
    log "campaign: $(echo "$todo" | wc -l | tr -d ' ') genome(s), run id $TWM_RUN_ID"
    # shellcheck disable=SC2086
    $RP campaign "$(_pod)" $todo
}

stage_poll() {
    local misses=0 rc=0 waited=0
    while :; do
        [ "$waited" -lt "$((POLL_MAX_MIN * 60))" ] \
            || die "poll: still running after ${POLL_MAX_MIN}m — refusing to wait longer; the pod is UP"
        rc=0; _ssh "grep -q '$TWM_RUN_ID' $POD_REPO/cloud/podresults/.done" >/dev/null 2>&1 || rc=$?
        if [ "$rc" = 0 ]; then log "poll: this run's marker present"; return; fi
        [ "$rc" = 255 ] && { log "poll: unreachable (transient) — no evidence either way"; misses=0; sleep 30; waited=$((waited+30)); continue; }
        rc=0; _ssh "tmux has-session -t $TMUX_SESSION" >/dev/null 2>&1 || rc=$?
        if [ "$rc" = 0 ]; then
            misses=0
            log "poll: running — $(_ssh "ls $POD_REPO/cloud/podresults/*.json 2>/dev/null | wc -l" 2>/dev/null | tr -d ' ') record(s)"
        elif [ "$rc" = 255 ]; then
            log "poll: unreachable (transient)"; misses=0
        else
            misses=$((misses + 1))
            [ "$misses" -lt 3 ] || {
                _ssh "tail -40 $POD_REPO/cloud/podresults/campaign.log" >&2 || true
                die "reachable, no session, no marker after 3 polls — a job died. ATTACH AND LOOK before re-running: a full re-run bootstraps (rsync --delete + uv sync) against a box that may still be working."
            }
        fi
        sleep 60; waited=$((waited+60))
    done
}

stage_download() {
    # Every pod writes cloud/podresults/campaign.log, and pull rsyncs into one local directory,
    # so a second lane's pull would silently replace the first lane's log. The per-genome records
    # have distinct names and do not collide; only the log does. Keep both.
    local id; id="$(_pod)"
    if [ -f cloud/podresults/campaign.log ]; then
        mkdir -p cloud/podresults-archive
        cp cloud/podresults/campaign.log \
           "cloud/podresults-archive/campaign-$(date -u +%Y%m%dT%H%M%SZ)-$id.log"
    fi
    log "download"
    $RP pull "$id"
    if [ -f cloud/podresults/campaign.log ]; then
        mkdir -p cloud/podresults-archive
        cp cloud/podresults/campaign.log \
           "cloud/podresults-archive/campaign-$(date -u +%Y%m%dT%H%M%SZ)-$id.log"
    fi
}

stage_verify() {
    local missing=0 failed=0 total=0 g f st
    for g in "$@"; do
        total=$((total+1)); f="$(_result_for "$g")"; st="$(_record_state "$f")"
        case "$st" in
            ok)     log "verify: ok $(basename "$f")" ;;
            failed) log "verify: FAILED-CANDIDATE $(basename "$f") (measured, ingestable)"; failed=$((failed+1)) ;;
            *)      log "verify: MISSING/unusable $f"; missing=1 ;;
        esac
    done
    [ "$missing" = 0 ] || die "results are not on local disk — do NOT terminate; run STAGE=download"
    if [ "$failed" = "$total" ] && [ "${LANE_ALLOW_ALL_FAILED:-0}" != 1 ]; then
        die "every record is a failure — that is the environment, not the candidates. Pod left up. LANE_ALLOW_ALL_FAILED=1 to override."
    fi
    log "verify: $total record(s) local, $failed failed"
}

stage_terminate() {
    stage_verify "$@"
    local id; id="$(_pod)"
    log "terminate: $id"
    $RP terminate "$id" || true
    # `runpod.sh terminate` pipes into jq with no curl -f, so it exits 0 on an API error and even
    # prints the word "terminated" for an auth failure. Confirm against the provider instead.
    local i lst
    for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
        lst="$($RP list 2>/dev/null || true)"
        if printf '%s' "$lst" | jq -e 'type == "array"' >/dev/null 2>&1; then
            printf '%s' "$lst" | jq -e --arg id "$id" 'any(.[]; .id == $id and .desiredStatus == "RUNNING")' >/dev/null 2>&1 \
                || { : > "$STATE"; log "terminate: CONFIRMED stopped"; return; }
        fi
        sleep 5
    done
    die "pod $id did NOT confirm termination — stop it now: cloud/runpod.sh terminate $id ; then cloud/runpod.sh reap"
}

cmd_run() {
    [ "$#" -gt 0 ] || die "usage: cloud/lane.sh run <genome.json>..."
    LANE_ARGS="$*"
    trap '_on_exit $?' EXIT
    : "${TWM_RUN_ID:=lane-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
    export TWM_RUN_ID
    if [ -n "${STAGE:-}" ]; then "stage_$STAGE" "$@"; return; fi
    # Deciding once, up front, stops provision and campaign disagreeing about whether there is
    # work — which otherwise rents a box, gives it nothing, and strands it.
    local todo; todo="$(_pending "$@")"
    if [ -z "$todo" ]; then
        if [ -s "$STATE" ] || [ -n "$POD" ]; then
            log "nothing pending; verifying and stopping the recorded pod"
            stage_terminate "$@"; return
        fi
        log "nothing pending and no pod recorded — nothing to do (FORCE=1 to re-measure)"; return
    fi
    for s in provision bootstrap campaign poll download verify terminate; do "stage_$s" "$@"; done
}

case "${1:-}" in
    run) shift; cmd_run "$@" ;;
    *) sed -n '2,28p' "$0"; exit 1 ;;
esac
