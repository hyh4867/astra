"""Download only a small, explicitly selected public LEVIR-CD sample archive.

Expected download: about 18 MB compressed (three image pairs + labels). Verify the
actual URL and archive contents before running; this script intentionally does not
fetch the full dataset.
"""
from pathlib import Path
from urllib.request import urlretrieve

# Public project sample archive; replace only with a verified, documented public URL.
SAMPLE_URL = "https://github.com/justchenhao/BIT_CD/releases/download/v1.0/LEVIR-CD-samples.zip"
DEST = Path(__file__).resolve().parents[1] / "data" / "levir_samples.zip"
if __name__ == "__main__":
    DEST.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(SAMPLE_URL, DEST)
    print(f"Saved {DEST} ({DEST.stat().st_size / 1024 / 1024:.1f} MB)")
