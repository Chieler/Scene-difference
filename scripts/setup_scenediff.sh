#!/bin/bash
# Fetches what scenediff_mps/ (official.py) needs but git does not carry: Pi3 and the SAM fork at
# pinned commits with our MPS patches, the SAM ViT-H checkpoint (2.4 GB), and the Pi3 and DINOv2
# weights (HF cache). Afterwards official.py runs with no network. Safe to rerun.
# Usage: uv sync && bash scripts/setup_scenediff.sh
set -euo pipefail
cd "$(dirname "$0")/../scenediff_mps"

fetch() {  # fetch <url> <commit> <dir> <patch>: shallow clone of one commit, then our patch
    local url=$1 commit=$2 dir=submodules/$3 patch=patches/$4
    if [ -f "$dir/.mps_patched" ]; then echo "$dir: ok"; return; fi
    rm -rf "$dir" && mkdir -p "$dir"
    git -C "$dir" init -q
    git -C "$dir" fetch -q --depth 1 "$url" "$commit"
    git -C "$dir" checkout -q FETCH_HEAD
    git -C "$dir" apply "../../$patch"
    touch "$dir/.mps_patched"
    echo "$dir: $commit + $patch"
}
fetch https://github.com/yyfz/Pi3 0a505af9c261a7c8e44946fba5b1942cdc25914b Pi3 Pi3.patch
fetch https://github.com/yuqunw/segment-anything-langsplat-modified 100f1cac9d57f74b5f8abcacb6daa2bf10e5cae4 \
    segment-anything-langsplat-modified segment-anything-langsplat-modified.patch

SAM=checkpoints/sam_vit_h_4b8939.pth
SAM_SHA=a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e
mkdir -p checkpoints
if [ ! -f "$SAM" ]; then
    curl -fL --retry 3 -C - -o "$SAM.part" https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    mv "$SAM.part" "$SAM"
fi
echo "$SAM_SHA  $SAM" | shasum -a 256 -c -

# The port loads the 'main' ref, so fetch that one: an offline from_pretrained resolves 'main' from
# the cache. Tested at yyfz233/Pi3 ae722e7, facebook/dinov2-large 47b73ee.
uv run python - <<'EOF'
from huggingface_hub import snapshot_download
for repo in ("yyfz233/Pi3", "facebook/dinov2-large"):
    snapshot_download(repo, allow_patterns=["config.json", "model.safetensors"])
    print(f"{repo}: ok")
EOF
