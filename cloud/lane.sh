#!/usr/bin/env bash
# One command from nothing to ingestable records, with the pod stopped at the end.
#
#   cloud/lane.sh run <genome.json>...     provision -> bootstrap -> campaign -> poll ->
#                                          download -> verify -> terminate
#   STAGE=<name> cloud/lane.sh run ...     run a single stage against the recorded pod
#   POD=<id>     cloud/lane.sh run ...     attach to a pod instead of provisioning one
#   FORCE=1      cloud/lane.sh run ...     re-measure genomes that already have a local
#                                          record. Resume skips them by name, which is what
#                                          you want after a crash and NOT what you want when
#                                          deliberately re-scoring the same ids.
#
# Why this exists rather than running the steps by hand. Two failures cost real money on this
# lane and neither is the kind a person reliably avoids by intending to:
#
#   1. A campaign finished and nothing noticed, so the box billed for hours doing nothing. A
#      watcher that infers completion from "the tmux session is gone" reports success when ssh
#      merely blinks; this one waits on a .done MARKER, treats an unreachable host as no
#      evidence at all, and needs three consecutive clean-but-sessionless polls to call failure.
#
#   2. The pod was terminated before the results were folded and copied home, which threw away a
#      measurement that had already been paid for. Here `verify` GATES `terminate`: every genome
#      must have a record on local disk carrying a fitness before the box can be stopped, and the
#      ERR trap deliberately leaves a pod RUNNING when anything fails, because an idle box costs
#      cents an hour and a lost campaign costs the campaign.
#
# Resumable: a genome whose record is already local is not re-run, so fix-and-rerun picks up
# where it stopped.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
RP="bash cloud/runpod.sh"
STATE="cloud/.lane-pod"
POD="${POD:-}"
: "${RUNPOD_ENV_TAG:=runpod-4090}"
export RUNPOD_ENV_TAG

log() { printf '[lane %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { printf '[lane] ERROR: %s\n' "$*" >&2; exit 1; }

# runpod.sh only auto-discovers ~/.ssh/id_ed25519 and ~/.ssh/id_rsa. A machine whose keys are
# named anything else gets a deploy that fails AFTER the confirmation, so resolve it here and
# fail before anything is billed.
if [ -z "${RUNPOD_SSH_KEY:-}" ]; then
    for c in "$HOME/.ssh/id_ed25519" "$HOME/.ssh/id_rsa" "$HOME/.ssh/lambda_cloud_ed25519"; do
        [ -f "$c" ] && [ -f "$c.pub" ] && { RUNPOD_SSH_KEY="$c"; break; }
    done
fi
[ -n "${RUNPOD_SSH_KEY:-}" ] || die "no ssh key with a matching .pub — set RUNPOD_SSH_KEY=<path to private key>"
export RUNPOD_SSH_KEY
log "ssh key: $RUNPOD_SSH_KEY"

_pod() {
    [ -n "$POD" ] && { echo "$POD"; return; }
    [ -s "$STATE" ] || die "no pod recorded in $STATE — run the provision stage, or pass POD=<id>"
    cat "$STATE"
}

_host() {
    local h; h="$($RP host "$(_pod)" 2>/dev/null | tail -1)"
    [ -n "$h" ] || return 1
    echo "$h"
}

_ssh() {
    local h ip port; h="$(_host)" || return 255
    ip="${h%% *}"; port="${h##* }"
    ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=25 \
        ${RUNPOD_SSH_KEY:+-i "$RUNPOD_SSH_KEY"} -p "$port" "root@$ip" "$@"
}

_result_for() { echo "cloud/podresults/$(basename "$1" .json).json"; }

_pending() {   # genomes with no verified local record yet
    local g; for g in "$@"; do
        local f; f="$(_result_for "$g")"
        if [ "${FORCE:-0}" != "1" ] && [ -s "$f" ] && grep -q '"fitness"' "$f"; then continue; fi
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
    [ -n "$todo" ] || { log "campaign: every genome already has a local record"; return; }
    log "campaign: $(echo "$todo" | wc -l | tr -d ' ') genome(s)"
    # shellcheck disable=SC2086
    $RP campaign "$(_pod)" $todo
}

stage_poll() {
    local misses=0
    while :; do
        if _ssh 'test -f ~/terminal-wm/cloud/podresults/.done' 2>/dev/null; then
            log "poll: .done present"; return
        elif ! _ssh true 2>/dev/null; then
            # An unreachable host is NOT evidence the run ended. This is the exact mistake that
            # once reported a campaign "finished" one minute in, and again on a network blip.
            log "poll: host unreachable (transient) — retry in 30s"; sleep 30; continue
        elif _ssh 'tmux has-session -t twm-campaign' 2>/dev/null; then
            misses=0
            log "poll: running — $(_ssh 'ls ~/terminal-wm/cloud/podresults/*.json 2>/dev/null | wc -l' | tr -d ' ') record(s) so far"
        else
            misses=$((misses + 1))
            [ "$misses" -lt 3 ] || {
                _ssh 'tail -40 ~/terminal-wm/cloud/podresults/campaign.log' >&2 || true
                die "reachable, no session, no .done after 3 polls — a job died. Fix and re-run; finished genomes are skipped."
            }
            sleep 30; continue
        fi
        sleep 60
    done
}

stage_download() { log "download"; $RP pull "$(_pod)"; }

stage_verify() {
    local missing=0 g f
    for g in "$@"; do
        f="$(_result_for "$g")"
        if [ -s "$f" ] && grep -q '"fitness"' "$f"; then
            log "verify: ok $(basename "$f")"
        else
            log "verify: MISSING or invalid $f"; missing=1
        fi
    done
    [ "$missing" = "0" ] || die "refusing to continue: results are not on local disk"
    log "verify: all $# record(s) local"
}

stage_terminate() {
    stage_verify "$@"          # the gate: never stop a box whose output is not home
    log "terminate: $(_pod)"
    $RP terminate "$(_pod)"
    : > "$STATE"
    log "terminate: done — ingest with 'evolve ingest --env $RUNPOD_ENV_TAG ...'"
}

cmd_run() {
    [ "$#" -gt 0 ] || die "usage: cloud/lane.sh run <genome.json>..."
    trap 'echo "[lane] FAILED — pod left RUNNING on purpose so nothing measured is lost. Inspect, then: STAGE=terminate cloud/lane.sh run $*" >&2' ERR
    if [ -n "${STAGE:-}" ]; then "stage_$STAGE" "$@"; trap - ERR; return; fi
    for s in provision bootstrap campaign poll download verify terminate; do "stage_$s" "$@"; done
    trap - ERR
}

case "${1:-}" in
    run) shift; cmd_run "$@" ;;
    *) sed -n '2,30p' "$0"; exit 1 ;;
esac
