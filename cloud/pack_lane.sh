#!/usr/bin/env bash
# The pack lane — everything GPU-heavy, on a rented box.
#
#   pack_lane.sh prepare                     one-time: pull the eye + the pack root, pin, encode
#   pack_lane.sh score <genome.json> <id>    train + measure a candidate, emit an ingestable result
#
# WHAT THIS COSTS YOU, STATED PLAINLY
# There are two supported ways to score on a remote box. Running the evolve CLI ON the box keeps
# every guarantee the engine offers: isolated-export scoring, the structural guard, novelty dedup,
# and the candidate-payload check. This script is the OTHER way — score here, carry the number
# back with `evolve ingest`. Ingest runs no guard, no dedup, no clone and no payload check, so the
# record it produces carries a score and not a reproducible artifact. Everything the engine would
# have enforced, you are enforcing by hand from here on. Prefer running the CLI on the box when
# you can; use this when you cannot.
set -euo pipefail

# MANDATORY thread caps. A rented GPU box reports the SHARED HOST's core count, while the
# container is CFS-quota-capped to a fraction of it. Sizing BLAS threads to the reported count
# measured as heavy throttling and once wedged a box badly enough to lose the
# session. Do not remove these.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

: "${TWM_DATA:=$REPO/data}"
: "${TWM_ENC:=$REPO/enc}"
: "${TWM_ARM:=treat}"                   # treat = the capability pack; control = the inert twin
: "${TWM_ENV_TAG:?set TWM_ENV_TAG to the environment name you will ingest under, e.g. runpod-4090}"

RAW="$TWM_DATA/dockerfs3-cupsF-$TWM_ARM"
ENC_ROOT="${RAW}-nocwd"
EYE="$TWM_ENC/e5-nocwd"

# The eye IS the frame. This sha is the frozen identity of the encoder every window in the scored
# root was embedded with; a different checkpoint silently redefines every embedding in the run.
EYE_TREE_SHA=b0379d30034f9f2a9359ba5c9c7b5933b06822f3982e06d5857ecfb0c15d6fd1

export TJ_FT_ENCODER="$EYE"
export TWM_CUPS_ROOT="$ENC_ROOT"
export TWM_EYE=enc_e5_ft_nocwd_hf
export TWM_EYE_TREE_SHA="$EYE_TREE_SHA"
export TWM_PYTHON="${TWM_PYTHON:-uv run python}"

say() { echo "[$(date +%H:%M:%S)] $*"; }

prepare() {
  mkdir -p "$TWM_DATA" "$TWM_ENC"

  say "pulling the eye"
  uv run python - <<PY
from huggingface_hub import snapshot_download
snapshot_download('veryfansome/terminal-jepa-encoders', allow_patterns=['e5-nocwd/*'],
                  local_dir='$TWM_ENC')
PY

  say "pinning the eye"
  uv run python - <<PY
from evolve import reencode as RE
got = RE._checkpoint_tree_sha('$EYE')
assert got == '$EYE_TREE_SHA', f"eye tree sha {got} != frozen pin $EYE_TREE_SHA"
print("eye ok")
PY

  say "pulling the raw pack root ($TWM_ARM)"
  uv run python - <<PY
from huggingface_hub import snapshot_download
snapshot_download('veryfansome/terminal-jepa-dockerfs', repo_type='dataset',
                  allow_patterns=['dockerfs3-cupsF-$TWM_ARM/*'], local_dir='$TWM_DATA')
PY

  if [ -f "$ENC_ROOT/emb-seq-val.pt" ]; then
    say "encoded root already present at $ENC_ROOT — skipping encode"
  else
    say "encoding cwd-out (one time, ~minutes on a GPU)"
    uv run python -m evolve.reencode --perception "$TWM_EYE" --src "$RAW" --out "$ENC_ROOT"
  fi

  say "preflight + embedding sha"
  uv run python -m eval.preflight
  say "PREPARE OK — $ENC_ROOT is scoreable. Pin the embedding_sha above as TWM_ROOT_SHA."
}

publish() {
  # Encode ONCE, publish, let every other box PULL the same bytes. Re-encoding per box is cheap in
  # wall-clock but not free in meaning: the eye is pinned, the tensors it produces are not, and a
  # forward pass on different hardware can differ in the last bits. That is a different
  # standardization frame with every existing check still passing.
  local repo="${TWM_HF_DATASET:-veryfansome/terminal-jepa-dockerfs}"
  say "publishing $ENC_ROOT to $repo"
  uv run python -m cloud.publish_root "$ENC_ROOT" "$repo"
}

score() {
  local genome="$1" cand_id="$2"
  local seeds="${TWM_SEEDS:-0,1,2}"
  local out="cloud/podresults/${cand_id}.json"
  export REPO_RESULTS="$REPO/.results/$cand_id"
  mkdir -p cloud/podresults "$REPO_RESULTS"

  IFS=',' read -ra SEEDLIST <<< "$seeds"
  for s in "${SEEDLIST[@]}"; do
    say "seed $s"
    $TWM_PYTHON -m eval.adapter "$REPO_RESULTS/s$s" "$genome" "$s" inner full
  done

  # Fold the per-seed metrics into one ingestable record. The engine means the per-seed scores
  # itself when it drives the eval; here we are outside it, so we do the same arithmetic and say
  # so explicitly. A seed that failed makes the whole candidate a failure — matching the engine's
  # all-or-nothing rule rather than quietly averaging over the survivors.
  uv run python - "$cand_id" "$out" "${SEEDLIST[@]}" <<'PY'
import json, os, pathlib, sys
cand, out = sys.argv[1], sys.argv[2]
seeds = [int(s) for s in sys.argv[3:]]
base = pathlib.Path(os.environ["REPO_RESULTS"])
per, pub, priv, fb = [], {}, {}, []
for s in seeds:
    m = json.loads((base / f"s{s}" / "metrics.json").read_text())
    if not m.get("correct") or m.get("combined_score") is None:
        json.dump({"fitness": None, "correct": False, "env": os.environ["TWM_ENV_TAG"],
                   "seeds": seeds, "guardrail": m.get("error", "seed_failed"),
                   "text_feedback": m.get("text_feedback", "")}, open(out, "w"), indent=1)
        print(f"CANDIDATE FAILED on seed {s}: {m.get('error')}")
        raise SystemExit(0)
    per.append(round(float(m["combined_score"]), 6))
    pub = m.get("public", {})                       # last seed wins, as the engine does
    priv[f"seed{s}"] = m.get("private", {})
    fb.append(m.get("text_feedback", ""))
json.dump({"fitness": round(sum(per) / len(per), 6), "correct": True,
           "env": os.environ["TWM_ENV_TAG"], "seeds": seeds, "per_seed": per,
           "public": pub, "private": priv, "text_feedback": "\n".join(fb)},
          open(out, "w"), indent=1)
print(json.dumps({"id": cand, "fitness": round(sum(per) / len(per), 6), "per_seed": per}))
PY

  say "wrote $out"
  echo
  echo "Bring it back with (from a Claude Code session in this repo):"
  echo "  evolve ingest --result $out --id $cand_id --genome $genome --mode full --split inner --env $TWM_ENV_TAG"
  echo
  echo "Before trusting these numbers alongside any scored elsewhere, run"
  echo "  evolve doctor --measure-env-offset"
  echo "so comparability is measured rather than assumed."
}

case "${1:-}" in
  prepare) prepare ;;
  publish) publish ;;
  score)   score "$2" "$3" ;;
  *) echo "usage: pack_lane.sh prepare | publish | score <genome.json> <candidate-id>" >&2
     exit 2 ;;
esac
