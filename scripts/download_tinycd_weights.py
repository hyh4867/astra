from pathlib import Path
from urllib.request import urlretrieve

DEST = Path(__file__).resolve().parents[1] / "models" / "tinycd_levir_best.pth"
URL = "https://raw.githubusercontent.com/AndreaCodegoni/Tiny_model_4_CD/main/pretrained_models/levir_best.pth"
MAX_BYTES = 5_000_000

if __name__ == "__main__":
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if not DEST.exists():
        urlretrieve(URL, DEST)
    size = DEST.stat().st_size
    if size > MAX_BYTES:
        DEST.unlink()
        raise ValueError(f"Unexpected checkpoint size: {size} bytes")
    print(f"TinyCD LEVIR checkpoint: {size:,} bytes at {DEST}")
