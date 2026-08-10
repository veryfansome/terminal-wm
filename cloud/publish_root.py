"""Publish an encoded pack root so every box pulls identical bytes instead of re-encoding.

    python -m cloud.publish_root <encoded-root> [<hf-dataset-repo>]

Prints the embedding sha to pin wherever the root is consumed; eval/preflight.py verifies it.
"""
import pathlib
import sys

from huggingface_hub import HfApi

from eval.preflight import embedding_sha

DEFAULT_REPO = "veryfansome/terminal-jepa-dockerfs"


def main(root, repo=DEFAULT_REPO):
    rp = pathlib.Path(root).resolve()
    for f in ("summary.json", "cache_meta.json", "emb-seq-train.pt", "emb-seq-val.pt"):
        if not (rp / f).exists():
            raise SystemExit(f"{rp} is missing {f} — this is not a complete encoded root")

    sha = embedding_sha(rp)
    print(f"publishing {rp.name} -> {repo}")
    HfApi().upload_folder(folder_path=str(rp), path_in_repo=rp.name,
                          repo_id=repo, repo_type="dataset")
    print("\npublished. Pin this everywhere the root is consumed:\n")
    print(f"  export TWM_ROOT_SHA={sha}\n")
    print("eval/preflight.py verifies it before any candidate runs, so a box that quietly "
          "re-encoded instead of pulling fails loud rather than scoring in its own frame.")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
