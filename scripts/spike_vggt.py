# scripts/spike_vggt.py
"""Throwaway: does VGGT load and run on MPS, and how many frames fit?"""
import sys, time, torch
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = VGGT.from_pretrained("facebook/VGGT-1B").to(dev).eval()
for s in map(int, sys.argv[1:] or ["16", "32", "48"]):
    x = torch.rand(s, 3, 294, 518, device=dev)
    t0 = time.time()
    with torch.no_grad():
        out = model(x)
    ext, intr = pose_encoding_to_extri_intri(out["pose_enc"], x.shape[-2:])
    print(s, "frames OK", f"{time.time() - t0:.1f}s", {k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)}, ext.shape, intr.shape)
