# diff.py
"""uv run diff.py A.mp4 B.mp4 -o out/"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np
from changes import Grid, carve, pair_moved, overlap, UNKNOWN, VOXEL
from recon import reconstruct, cloud, Views
import change2d, label
from viewer import write_viewer

FIELDS = ("images", "depth", "conf", "K", "E")  # Views fields, in order, for the recon cache
VOTE = (0.7, 0.7, 2, 20)  # change2d.vote thr, ratio, min_votes, min_voxels; best F1 in the SceneDiff val sweep
MIN_OVERLAP = 0.25  # below this the videos do not show the same space
MIN_SIM, VOL_RATIO = 0.85, 0.5  # pair_moved: CLIP similarity and volume ratio for a removed+added pair
FRAMES = 24  # per video; both videos go through VGGT in one pass, so 2 * FRAMES images (~119 s on MPS)


def cached_reconstruct(a, b, frames, out):
    """VGGT is the slow part; keep its output so reruns and eval sweeps skip it."""
    cache, key = out / "recon.npz", f"{a.resolve()}|{b.resolve()}|{frames}"
    try:
        z = np.load(cache)
        if str(z["key"]) == key:
            return tuple(Views(*(z[f"{s}{f}"] for f in FIELDS)) for s in "AB")
    except Exception:  # missing, or half-written by a run that died: recompute
        pass
    A, B = reconstruct(a, b, frames, out / "frames")
    tmp = out / "recon.tmp.npz"
    np.savez(tmp, key=key, **{f"{s}{f}": getattr(v, f) for s, v in (("A", A), ("B", B)) for f in FIELDS})
    tmp.rename(cache)  # atomic, so a crash never leaves a cache that looks valid
    return A, B


def detect(A, B):
    (pa, oa, ca), (pb, ob, cb) = cloud(A), cloud(B)
    grid = Grid.around(pa, pb)
    sa, sb = carve(grid, pa, oa), carve(grid, pb, ob)
    ov = overlap(sa, sb)
    if ov < MIN_OVERLAP:
        raise ValueError(f"videos don't overlap enough ({ov:.0%}), re-record with shared walls and floor in view")

    fa, fb = change2d.dino(A.images), change2d.dino(B.images)
    ba = change2d.vote(*change2d.scores(B, A, fb, fa), grid, "added", *VOTE)
    br = change2d.vote(*change2d.scores(A, B, fa, fb), grid, "removed", *VOTE)
    ea = label.embed([label.best_crop(b, B) for b in ba])
    er = label.embed([label.best_crop(b, A) for b in br])
    pairs = pair_moved(br, ba, er @ ea.T, MIN_SIM, VOL_RATIO) if len(ba) and len(br) else []
    lo, hi = grid.lo, grid.lo + np.array(grid.shape) * VOXEL
    place = lambda b: label.where(b.center, lo, hi)

    changes, moved_r, moved_a = [], {i for i, _ in pairs}, {j for _, j in pairs}
    box = lambda b: {"lo": b.lo.tolist(), "hi": b.hi.tolist()}
    for i, j in pairs:
        w, pr = label.name(ea[j] + er[i])
        changes.append({"type": "moved", "label": w, "score": pr, **box(ba[j]), "from_lo": br[i].lo.tolist(),
                        "from_hi": br[i].hi.tolist(), "text": label.sentence("moved", w, place(ba[j]), place(br[i]))})
    for kind, bl, em, skip in (("added", ba, ea, moved_a), ("removed", br, er, moved_r)):
        for k, b in enumerate(bl):
            if k not in skip:
                w, pr = label.name(em[k])
                changes.append({"type": kind, "label": w, "score": pr, **box(b), "from_lo": None, "from_hi": None,
                                "text": label.sentence(kind, w, place(b))})
    return changes, (grid, sa, br, pa, ca, pb, cb)


def fail(msg):
    sys.exit(f"error: {msg}")


def main():
    ap = argparse.ArgumentParser(description="Find what changed in a room between two videos.")
    ap.add_argument("a", type=Path); ap.add_argument("b", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("out"))
    ap.add_argument("--frames", type=int, default=FRAMES, help="frames sampled per video")
    args = ap.parse_args()
    for p in (args.a, args.b):
        if not p.is_file():
            fail(f"video not found: {p}")
    if shutil.which("ffmpeg") is None:
        fail("ffmpeg not found, install it first (brew install ffmpeg)")
    args.out.mkdir(parents=True, exist_ok=True)

    try:
        A, B = cached_reconstruct(args.a, args.b, args.frames, args.out)
        changes, (grid, sa, br, pa, ca, pb, cb) = detect(A, B)
    except (RuntimeError, ValueError) as e:
        fail(str(e))

    (args.out / "changes.json").write_text(json.dumps(changes, indent=2))
    np.savez(args.out / "cams.npz", KA=A.K, EA=A.E, KB=B.K, EB=B.E, hw=A.depth.shape[1:])  # for eval.py
    (args.out / "report.md").write_text("# What changed\n\n" + (
        "\n".join(f"- {c['text']} (confidence {c['score']:.0%})" for c in changes) or "No changes detected.") + "\n")

    unseen = sa[tuple(np.clip(grid.index(pb), 0, np.array(grid.shape) - 1).T)] == UNKNOWN
    gray = cb.copy(); gray[unseen] = 120
    ghost_mask = np.zeros(grid.shape, bool)
    for b in br:
        ghost_mask[tuple(b.idx.T)] = True
    gi = grid.index(pa); ok = grid.inside(gi)
    ghost = pa[ok][ghost_mask[tuple(gi[ok].T)]]
    write_viewer(args.out / "viewer.html", {"A": {"pos": pa, "col": ca}, "B": {"pos": pb, "col": cb, "gray": gray},
                                            "ghost": {"pos": ghost}, "changes": changes})
    print((args.out / "report.md").read_text())
    print(f"viewer: {args.out / 'viewer.html'}")


if __name__ == "__main__":
    main()
