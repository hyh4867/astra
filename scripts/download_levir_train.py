"""Download and verify the public LEVIR-CD training archive locally.

The archive is academic-use only and is intentionally Git-ignored.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import Request, urlopen


URL = "https://huggingface.co/datasets/satellite-image-deep-learning/LEVIR-CD/resolve/main/train.zip"
EXPECTED_BYTES = 1_721_956_862
EXPECTED_MD5 = "a638e71f480628652dea78d8544307e4"
DEST = Path(__file__).resolve().parents[1] / "data" / "raw"


def hashes(path: Path) -> tuple[int, str, str]:
    sha256, md5 = hashlib.sha256(), hashlib.md5()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            size += len(block)
            sha256.update(block)
            md5.update(block)
    return size, sha256.hexdigest(), md5.hexdigest()


def download() -> Path:
    DEST.mkdir(parents=True, exist_ok=True)
    target = DEST / "train.zip"
    part = DEST / "train.zip.part"
    offset = part.stat().st_size if part.exists() else 0
    request = Request(URL, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urlopen(request, timeout=90) as response:
        if offset and response.status != 206:
            offset = 0
        mode = "ab" if offset else "wb"
        last_log = offset // (64 * 1024 * 1024)
        with part.open(mode) as output:
            while block := response.read(4 * 1024 * 1024):
                output.write(block)
                bucket = part.stat().st_size // (64 * 1024 * 1024)
                if bucket > last_log:
                    last_log = bucket
                    print(f"downloaded {part.stat().st_size:,}/{EXPECTED_BYTES:,} bytes", flush=True)
    size, sha256, md5 = hashes(part)
    if size != EXPECTED_BYTES:
        raise RuntimeError(f"size mismatch: expected {EXPECTED_BYTES}, got {size}; partial kept for resume")
    if md5 != EXPECTED_MD5:
        raise RuntimeError(f"canonical MD5 mismatch: {md5}; partial kept for investigation")
    part.replace(target)
    print(f"verified train.zip bytes={size} md5={md5} sha256={sha256}", flush=True)
    return target


if __name__ == "__main__":
    download()
