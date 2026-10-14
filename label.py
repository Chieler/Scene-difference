# label.py
"""CLIP zero-shot naming of changed blobs, plus the report wording."""
import itertools
import numpy as np
from PIL import Image

VOCAB = """chair, armchair, sofa, couch, table, desk, coffee table, side table, stool, bench, bed, pillow, blanket,
cushion, shelf, bookshelf, cabinet, drawer, wardrobe, box, cardboard box, bag, backpack, suitcase, basket, bin,
trash can, lamp, floor lamp, monitor, laptop, keyboard, television, speaker, plant, vase, book, books, bottle,
cup, mug, plate, bowl, pot, shoes, jacket, clothes, towel, rug, carpet, mat, curtain, mirror, painting, clock,
toy, ball, guitar, fan, heater, vacuum cleaner, ladder, bicycle, person, dog, cat, cable, papers, whiteboard"""
VOCAB = [w.strip() for w in VOCAB.replace("\n", " ").split(",") if w.strip()]

_m = None
_txt = None


def _model():
    global _m, _txt
    if _m is None:
        from sentence_transformers import SentenceTransformer
        _m = SentenceTransformer("clip-ViT-L-14")  # B-32 called a clear backpack crop "vacuum cleaner"
        _txt = _m.encode([f"a photo of a {w}" for w in VOCAB], normalize_embeddings=True)
    return _m


def embed(images):
    m = _model()
    out = np.zeros((len(images), _txt.shape[1]), np.float32)
    ok = [i for i, im in enumerate(images) if im is not None]
    if ok:
        out[ok] = m.encode([images[i] for i in ok], normalize_embeddings=True)
    return out


def name(emb):
    _model()
    n = np.linalg.norm(emb)
    if n == 0:
        return "object", 0.0
    logits = 100 * _txt @ (emb / n)
    p = np.exp(logits - logits.max()); p /= p.sum()
    i = int(p.argmax())
    return VOCAB[i], float(p[i])


def best_crop(blob, views, pad=0.15):
    """Largest fully-visible, unoccluded view of the blob's bounding box."""
    corners = np.array(list(itertools.product(*zip(blob.lo, blob.hi))))
    tol = np.linalg.norm(blob.hi - blob.lo) / 2 + 0.1  # a surface nearer than this is in front of the blob
    best, best_area = None, 0
    for i in range(len(views.E)):
        R, t, K = views.E[i, :, :3], views.E[i, :, 3], views.K[i]
        cam, c = corners @ R.T + t, blob.center @ R.T + t
        if c[2] < 0.3 or (cam[:, 2] < 0.1).any():
            continue
        uv = (cam @ K.T); uv = uv[:, :2] / uv[:, 2:]
        (x0, y0), (x1, y1) = uv.min(0), uv.max(0)
        H, W = views.depth[i].shape
        if x0 < 0 or y0 < 0 or x1 >= W or y1 >= H or (x1 - x0) < 12 or (y1 - y0) < 12:
            continue
        u, v = (K @ c)[:2] / c[2]
        d = views.depth[i][int(v), int(u)]
        if d > 0 and d < c[2] - tol:  # occluded
            continue
        if (x1 - x0) * (y1 - y0) > best_area:
            px, py = pad * (x1 - x0), pad * (y1 - y0)
            best_area = (x1 - x0) * (y1 - y0)
            best = Image.fromarray(views.images[i]).crop(
                (max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py)))
    return best


def where(center, lo, hi):
    """Coarse place in the room. Axes are the reconstruction's own, so left/front are arbitrary but consistent."""
    f = (center[:2] - lo[:2]) / np.maximum(hi[:2] - lo[:2], 1e-6)
    pick = lambda x, names: names[min(2, max(0, int(x * 3)))]
    parts = [w for w in (pick(f[0], ("left", "center", "right")), pick(f[1], ("front", "middle", "back")))
             if w not in ("center", "middle")]
    return f"the {', '.join(parts)} of the room" if parts else "the middle of the room"


def sentence(kind, word, place, from_place=None):
    if kind == "moved":
        if from_place is None:
            raise ValueError("moved change needs from_place")
        return f"Moved: {word}, from {from_place} to {place}" if from_place != place else f"Moved: {word}, within {place}"
    return f"{kind.capitalize()}: {word} in {place}"
