# fusion.py
"""Candidates from the official SceneDiff method (official.py), turned into named object changes two ways:

- full:    fused with the VLM's own claims. Objects both found keep the VLM's name and sentence; objects only the
           official method found are shown to the VLM as outlined crops and named, or vetoed, by it (vlm.verify).
           On the SceneDiff benchmark this combination scores obj/im AP 0.585, against 0.548 and 0.322 for its parts.
- offline: no API. Objects are named by CLIP from their best mask crop.

Either way every object then goes through the same 3D check against the other video (semantic.check).
"""
import numpy as np
from PIL import Image
import label, official, semantic

MATCH_IOU = 0.3  # mean mask IoU over shared frames for "both methods found it"; as scripts/fuse.py


def kind(masks):
    vs = {v for v, _ in masks}
    return "moved" if vs == {"A", "B"} else ("removed" if vs == {"A"} else "added")


def iou(a, b):
    ious = [(a[k] & b[k]).sum() / max(1, (a[k] | b[k]).sum()) for k in a if k in b]
    return float(np.mean(ious)) if ious else 0.0


def candidates(pa, pb, out):
    """The official method's objects as {masks: {(video, frame): mask}, cost}."""
    return [{"masks": {(v, f): m for v in "AB" for f, m in c.get(v, {}).items() if m.any()}, "cost": float(c["cost"])}
            for c in official.candidates(pa, pb, out)]


def fuse(off, claims):
    """(agreed, official-only, VLM-only): official objects matched to a VLM claim take its name, sentence and
    confidence; the official mask wins in frames where both have one, the VLM's fills the rest."""
    agreed, off_only = {}, []  # claim index -> fused object; SceneDiff may split one object in several
    for o in off:
        j, best = max(((j, iou(o["masks"], c["masks"])) for j, c in enumerate(claims)), key=lambda t: t[1],
                      default=(None, 0.0))
        if best >= MATCH_IOU:
            prev = agreed[j]["masks"] if j in agreed else claims[j]["masks"]
            masks = {**prev, **{k: m | prev[k] if k in prev and j in agreed else m for k, m in o["masks"].items()}}
            agreed[j] = {**claims[j], "masks": masks, "change": kind(masks), "source": "both"}
        else:
            off_only.append(o)
    used, agreed = set(agreed), list(agreed.values())
    return agreed, off_only, [{**c, "source": "vlm"} for j, c in enumerate(claims) if j not in used]


def full(A, B, pa, pb, out):
    import vlm
    frames = {"A": pa, "B": pb}
    off = candidates(pa, pb, out)
    claims = semantic.ask_vlm(pa, pb, out / "vlm.json")
    claims = semantic.claim_objects(claims, semantic.segment(claims, frames)) if claims else []
    agreed, off_only, vlm_only = fuse(off, claims)
    answers = vlm.verify([{v: {f: m for (vv, f), m in o["masks"].items() if vv == v} for v in "AB"} for o in off_only],
                         frames, out / "verify.json") if off_only else {}
    checked = []
    for i, o in enumerate(off_only):
        a = answers.get(i, {"real": True, "name": "object", "description": "", "p_real": 0.5})
        checked.append({"name": a["name"], "description": a["description"], "change": kind(o["masks"]),
                        "masks": o["masks"], "score": 1 + min(a["p_real"], 0.999), "source": "official",
                        "veto": None if a["real"] else "the VLM sees no change here"})
    objects = ([{**o, "score": 2 + min(o["score"], 0.999)} for o in agreed] + checked +
               [{**o, "score": min(o["score"], 0.999)} for o in vlm_only])  # ranking bands as on the benchmark
    return semantic.check(objects, A, B, pa, pb, out)


def offline(A, B, pa, pb, out):
    frames = {"A": pa, "B": pb}
    off = candidates(pa, pb, out)
    crops = [crop(o["masks"], frames) for o in off]
    names = [label.name(e) for e in label.embed(crops)] if crops else []
    objects = [{"name": n, "description": "", "change": kind(o["masks"]), "masks": o["masks"],
                "score": min(o["cost"], 0.999), "source": "official"} for o, (n, _) in zip(off, names)]
    return semantic.check(objects, A, B, pa, pb, out)


def crop(masks, frames, pad=0.15):
    """The object's largest mask, cropped from its frame with a little context, for CLIP."""
    (v, f), m = max(masks.items(), key=lambda t: t[1].sum())
    im = Image.open(frames[v][f]).convert("RGB")
    ys, xs = np.nonzero(m)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    px, py = pad * (x1 - x0) + 4, pad * (y1 - y0) + 4
    return im.crop((max(0, x0 - px), max(0, y0 - py), min(im.width, x1 + px), min(im.height, y1 + py)))
