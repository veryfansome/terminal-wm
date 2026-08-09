"""The mandatory 3-way split (the anti-overfitting rail). All held-out (unseen) images:
  inner-val  = fedora + mariadb   -> fitness is scored here (the loop optimizes against it)
  final-test = rockylinux + httpd -> the loop NEVER scores these; only the champion, once.
Each pair mixes a distro + a service image so the two held-out sets are comparable."""

INNER_IMAGES = ("fedora", "mariadb")
TEST_IMAGES = ("rocky", "httpd")


def _match(image, keys):
    return any(k in image for k in keys)


def split_val(val_seqs, which="inner"):
    keys = INNER_IMAGES if which == "inner" else TEST_IMAGES
    out = [s for s in val_seqs if _match(s["image"], keys)]
    if not out:
        raise ValueError(f"no val sequences matched {which} images {keys}; "
                         f"images present: {sorted({s['image'] for s in val_seqs})}")
    return out
