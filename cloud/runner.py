"""Concurrent (genome, seed) runner for the pack lane.

    python -m cloud.runner --genomes evolve/genomes/*.json --seeds 0,1,2 --concurrency 6 --gpus 4

The scoring engine is deliberately synchronous and has no job queue, no parallelism and no GPU
arbitration — it scores one candidate at a time. That is fine for the engine's own bookkeeping and
useless for a measurement campaign, where the work is dozens of independent (genome, seed) trains
that should saturate a rented box. So the pack lane runs them here and the numbers go back through
the engine's ingest path afterwards.

THE JOB IS SMALL AND THE GPU IS THE LIMIT
  A single train on this world model is kernel-launch bound, not memory bound — one GPU is not
  saturated by one job and is saturated by roughly three. So throughput scales with GPU COUNT, and
  concurrency should be about three per visible GPU. Running one job at a time wastes most of the
  box; running twenty wastes the queue.

THREAD CAPS ARE MANDATORY, AND nproc IS A LIE
  A container reports the SHARED HOST's core count while being CFS-quota-capped to a fraction of
  it. Sizing BLAS threads to the reported count is how you get a box pegged at heavy throttling
  with the GPU idle. This module reads the real cgroup quota and divides it across the concurrent
  jobs. Measured consequence of getting this wrong, once: a box wedged badly enough to lose the
  session.

Each job writes a .done sentinel, so re-running the command resumes rather than repeating. One job
failing does not take down the batch — its reason is recorded and the rest continue, because a
failed candidate is a result.
"""
import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import subprocess
import sys
import time


def cpu_quota():
    """The container's REAL cpu allowance, not the host's core count."""
    v2 = pathlib.Path("/sys/fs/cgroup/cpu.max")
    if v2.exists():
        quota, period = (v2.read_text().split() + ["100000"])[:2]
        if quota != "max":
            return max(1, int(float(quota) / float(period)))
    q = pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    p = pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if q.exists() and p.exists():
        qv, pv = int(q.read_text()), int(p.read_text())
        if qv > 0:
            return max(1, qv // pv)
    return os.cpu_count() or 4


def run_job(genome, seed, gpu, threads, outdir, split, mode, timeout_s):
    tag = f"{pathlib.Path(genome).stem}.s{seed}"
    d = pathlib.Path(outdir) / pathlib.Path(genome).stem / f"s{seed}"
    if (d / ".done").exists():
        return {"job": tag, "status": "cached"}
    d.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = str(threads)
    py = env.get("TWM_PYTHON", sys.executable).split()

    t0 = time.time()
    try:
        r = subprocess.run(py + ["-m", "eval.adapter", str(d), genome, str(seed), split, mode],
                           env=env, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        (d / "error.txt").write_text(f"timeout after {timeout_s}s")
        return {"job": tag, "status": "timeout", "secs": round(time.time() - t0)}

    if r.returncode != 0 or not (d / "metrics.json").exists():
        (d / "error.txt").write_text((r.stdout or "") + "\n" + (r.stderr or ""))
        # An environment fault is not a candidate fault. Say which, loudly, so a broken box does
        # not get written up as a population of bad candidates.
        kind = "infra" if "PREFLIGHT FAILED" in (r.stderr or "") else "failed"
        return {"job": tag, "status": kind, "secs": round(time.time() - t0),
                "detail": (r.stderr or "")[-300:]}

    (d / ".done").write_text("ok")
    m = json.loads((d / "metrics.json").read_text())
    return {"job": tag, "status": "ok", "secs": round(time.time() - t0),
            "score": m.get("combined_score"), "correct": m.get("correct")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--genomes", nargs="+", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=None,
                    help="default: 3 per visible GPU — the measured saturation point")
    ap.add_argument("--out", default=".results")
    ap.add_argument("--split", default="inner")
    ap.add_argument("--mode", default="full")
    ap.add_argument("--timeout-s", type=int, default=7200)
    a = ap.parse_args(argv)

    seeds = [int(s) for s in a.seeds.split(",")]
    conc = a.concurrency or (3 * max(1, a.gpus))

    # Derive the lane ONCE. The standardized splits, the window layout and the role-swap chains
    # are properties of (root, split, eye, swap-seed), not of any genome, and synthesizing them
    # per worker means reloading the text encoder and re-encoding every alternative chain N times
    # for a bit-identical result. Workers memory-map this, so they share one physical copy.
    if not os.environ.get("TWM_CONTEXT"):
        ctx_path = pathlib.Path(a.out) / f"lane-{a.split}.pt"
        if not ctx_path.exists():
            print(f"building shared lane context -> {ctx_path}", flush=True)
            from cloud import build_context
            if build_context.main(["--split", a.split, "--out", str(ctx_path)]) != 0:
                return 2
        os.environ["TWM_CONTEXT"] = str(ctx_path)
    print(f"shared lane context: {os.environ['TWM_CONTEXT']}", flush=True)
    threads = max(1, cpu_quota() // conc)
    jobs = [(g, s) for g in a.genomes for s in seeds]

    print(f"{len(jobs)} jobs | concurrency {conc} over {a.gpus} gpu(s) | "
          f"cpu quota {cpu_quota()} -> {threads} thread(s)/job", flush=True)

    results = []
    with cf.ThreadPoolExecutor(max_workers=conc) as ex:
        futs = {ex.submit(run_job, g, s, i % max(1, a.gpus), threads,
                          a.out, a.split, a.mode, a.timeout_s): (g, s)
                for i, (g, s) in enumerate(jobs)}
        for f in cf.as_completed(futs):
            r = f.result()
            results.append(r)
            print(json.dumps(r), flush=True)

    by = {}
    for r in results:
        by[r["status"]] = by.get(r["status"], 0) + 1
    print("\n" + json.dumps({"summary": by}, indent=1))
    if by.get("infra"):
        print("\nINFRA failures present — the box is wrong, not the candidates. Fix and re-run; "
              "completed jobs are cached.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
