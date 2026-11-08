"""Download LEVIR-CD val/test only from a Hugging Face mirror.

The original dataset is academic-use only; see README. The 1.72 GB train split
is deliberately excluded. Resume partial downloads and enforce a 1 GB file cap.
"""
from pathlib import Path
from urllib.request import Request, urlopen

BASE = "https://huggingface.co/datasets/satellite-image-deep-learning/LEVIR-CD/resolve/main/"
FILES = {"val.zip": 246_152_048, "test.zip": 496_305_323}
MAX_FILE_BYTES = 1_000_000_000
DEST = Path(__file__).resolve().parents[1] / "data" / "raw"


def download(name: str, expected: int):
    if expected > MAX_FILE_BYTES:
        raise ValueError(f"Refusing file larger than 1 GB: {name}")
    path = DEST / name
    part = path.with_suffix(path.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    request = Request(BASE + name, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urlopen(request, timeout=60) as response:
        # Some mirrors ignore Range. Restart rather than append a full response.
        if offset and response.status != 206:
            offset = 0
        with part.open("ab" if offset else "wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
    actual = part.stat().st_size
    if actual != expected:
        raise IOError(f"{name}: expected {expected:,} bytes, got {actual:,}; rerun to resume")
    part.replace(path)
    print(f"Saved {path} ({actual:,} bytes)", flush=True)


if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    for filename, size in FILES.items():
        if (DEST / filename).exists() and (DEST / filename).stat().st_size == size:
            print(f"Already present: {filename}")
        else:
            download(filename, size)
