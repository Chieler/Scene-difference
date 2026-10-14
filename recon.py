"""Videos -> one shared, gravity-up, metric-scale reconstruction (VGGT)."""
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image

CAM_HEIGHT = 1.4  # assumed handheld camera height in meters; sets the metric scale
LONG_SIDE = 518


@dataclass
class Views:
    images: np.ndarray
    depth: np.ndarray
    conf: np.ndarray
    K: np.ndarray
    E: np.ndarray

    def part(self, a, b):
        return Views(*(x[a:b] for x in (self.images, self.depth, self.conf, self.K, self.E)))


def centers(v):
    return -np.einsum("sji,sj->si", v.E[:, :, :3], v.E[:, :, 3])


def unproject(depth, K, E):
    H, W = depth.shape
    vv, uu = np.mgrid[:H, :W]
    cam = np.stack([(uu - K[0, 2]) / K[0, 0] * depth, (vv - K[1, 2]) / K[1, 1] * depth, depth], -1)
    return (cam - E[:, 3]) @ E[:, :3]  # R^T (cam - t)


def cloud(v, conf_pct=40, stride=3, max_pts=150_000):
    S, H, W = v.depth.shape
    P = np.stack([unproject(v.depth[i], v.K[i], v.E[i]) for i in range(S)])
    O = np.broadcast_to(centers(v)[:, None, None, :], P.shape)
    keep = (v.conf >= np.percentile(v.conf, conf_pct)) & (v.depth > 0)
    sub = np.zeros_like(keep); sub[:, ::stride, ::stride] = True
    keep &= sub
    P, O, C = P[keep], O[keep], v.images[keep]
    if len(P) > max_pts:
        sel = np.random.default_rng(0).choice(len(P), max_pts, replace=False)
        P, O, C = P[sel], O[sel], C[sel]
    return P.astype(np.float32), O.astype(np.float32), C


def rot_between(a, b):
    """Rotation matrix taking unit vector a to unit vector b."""
    v, c = np.cross(a, b), float(a @ b)
    if c < -0.999999:  # anti-parallel: 180 degrees about any axis perpendicular to a
        ax = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(ax) < 1e-6:
            ax = np.array([0.0, 1.0, 0.0])
        ax /= np.linalg.norm(ax)
        return 2 * np.outer(ax, ax) - np.eye(3)
    k =np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def floor_transform(points, cams, ups, iters=400, seed=0):
    """RANSAC the largest near-horizontal plane (horizontal = normal close to mean camera up)."""
    rng = np.random.default_rng(seed)
    up0 = ups.mean(0); up0 /= np.linalg.norm(up0)
    pts = points[rng.choice(len(points), min(len(points), 30000), replace=False)]
    thr = 0.02 * np.median(np.linalg.norm(pts[:, None] - cams[None], axis=2))  # scale-free tolerance
    best, best_n = None, 0
    for _ in range(iters):
        p = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if abs(n @ up0) < 0.9:  # more than ~25 degrees from horizontal: a wall, skip
            continue
        if (np.median(cams, axis=0) - p[0]) @ (n if n @ up0 > 0 else -n) <= 0:  # cameras below plane: a ceiling
            continue
        inl = np.abs((pts - p[0]) @ n) < thr
        if inl.sum() > best_n:
            best, best_n = inl, inl.sum()
    if best is None or best_n < 0.05 * len(pts):
        raise RuntimeError("no floor found: keep the floor in view in both videos")
    q = pts[best]; p0 = q.mean(0)
    n = np.linalg.svd(q - p0)[2][-1]
    if n @ up0 < 0:
        n = -n
    h = np.median((cams - p0) @ n)
    if h <= 0:
        raise RuntimeError("cameras are below the detected floor: floor detection failed")
    return CAM_HEIGHT / h, rot_between(n, np.array([0.0, 0.0, 1.0])), p0


def align(v, s, R, p0):
    Rc = v.E[:, :, :3]
    t = s * (v.E[:, :, 3] + np.einsum("sij,j->si", Rc, p0))
    E = np.concatenate([Rc @ R.T, t[:, :, None]], axis=2)
    return Views(v.images, v.depth * s, v.conf, v.K, E)


def extract_frames(video, n, out):
    shutil.rmtree(out, ignore_errors=True)  # glob below would pick up stale frames from a run with more frames
    out.mkdir(parents=True, exist_ok=True)
    try:
        dur = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, check=True).stdout)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", f"fps={n / dur}",
                        "-frames:v", str(n), str(out / "%03d.jpg")], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        lines = e.stderr.strip().splitlines() or ["ffmpeg/ffprobe failed"]
        raise RuntimeError(lines[-1]) from e
    paths = sorted(out.glob("*.jpg"))
    if len(paths) < 4:
        raise RuntimeError(f"{video}: could not extract enough frames")
    return paths


def load_frames(paths, size):
    ims = [np.asarray(Image.open(p).convert("RGB").resize(size, Image.BICUBIC)) for p in paths]
    return np.stack(ims)


def frame_size(path):
    w, h = Image.open(path).size
    s = LONG_SIDE / max(w, h)
    return tuple(max(14, round(d * s / 14) * 14) for d in (w, h))  # VGGT patch size is 14


def run_vggt(images):
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(dev).eval()
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float().div(255).to(dev)
    with torch.no_grad():
        out = model(x)
    E, K = pose_encoding_to_extri_intri(out["pose_enc"], x.shape[-2:])
    g = lambda t: t[0].float().cpu().numpy()
    return g(out["depth"])[..., 0], g(out["depth_conf"]), g(K), g(E)


def reconstruct(video_a, video_b, frames, work_dir):
    pa = extract_frames(video_a, frames, work_dir / "A")
    pb = extract_frames(video_b, frames, work_dir / "B")
    sa, sb = frame_size(pa[0]), frame_size(pb[0])
    if sa != sb:
        raise RuntimeError(f"videos have different orientation or aspect ({sa} vs {sb}): record both the same way")
    imgs = load_frames(pa + pb, sa)
    depth, conf, K, E = run_vggt(imgs)
    v = Views(imgs, depth, conf, K, E)
    pts, _, _ = cloud(v)
    ups = -E[:, 1, :3]  # camera y points down, so up = -row 1 of R
    s, R, p0 = floor_transform(pts, centers(v), ups)
    v = align(v, s, R, p0)
    return v.part(0, len(pa)), v.part(len(pa), len(pa) + len(pb))
