"""Pull single pairs out of the 12 GB SceneDiff zip with HTTP range reads.

    python scripts/fetch_scenediff.py --list
    python scripts/fetch_scenediff.py bedroom_1_bedroom_2 office_1_office_2 -o data/scenediff
"""
import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://huggingface.co/datasets/yuqun/SceneDiff/resolve/main/scenediff_benchmark.zip"
KEEP = ("original_video", "segments.pkl", "review_meta.json")  # skip overlay/compressed copies


class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.url = r.url  # follow the CDN redirect once
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = [off, self.pos + off, self.size + off][whence]
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pairs", nargs="*")
    p.add_argument("--list", action="store_true")
    p.add_argument("-o", "--out", type=Path, default=Path("data/scenediff"))
    a = p.parse_args()
    # big buffer so zipfile's small reads don't each become an HTTP request
    z = zipfile.ZipFile(io.BufferedReader(HttpFile(URL), 8 << 20))
    names = z.namelist()
    if a.list or not a.pairs:
        print("\n".join(sorted({n.split("/")[2] for n in names if n.count("/") > 2 and "/data/" in n})))
        return
    for n in names:
        parts = n.split("/")
        if len(parts) == 4 and parts[2] in a.pairs and any(k in parts[3] for k in KEEP) \
                or parts[1] == "splits" and parts[-1]:
            dest = a.out / n
            if dest.exists():
                continue
            print(f"{z.getinfo(n).file_size / 1e6:8.1f} MB  {n}", flush=True)
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_name(dest.name + ".part")  # a dropped connection must not leave a "done" file
            with z.open(n) as src, open(part, "wb") as f:
                while chunk := src.read(8 << 20):
                    f.write(chunk)
            part.rename(dest)


if __name__ == "__main__":
    main()
