"""Fetch the TinyCD model code that ASTRA imports.

TinyCD (https://github.com/AndreaCodegoni/Tiny_model_4_CD) is released by its
authors for non-commercial and research purposes only, without an open-source
license, so it is not included in this repository. This script downloads the
three model files from a pinned upstream commit, checks their SHA-256 and
writes them to vendor/tinycd/models/, which Git ignores. It does not download
datasets or model weights.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from urllib.request import urlopen

UPSTREAM = "https://raw.githubusercontent.com/AndreaCodegoni/Tiny_model_4_CD"
COMMIT = "1696827ffb7473c448591aaaa4434d4d4934a1ba"
FILES = {
    "__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "change_classifier.py": "b9ec63a09859afe48617f4ffcb2a71990b4627bad082e39cc71e24e5ba291eab",
    "layers.py": "5fcfef498f68b53e069722d3d98dc6019f44fca2f59dc926d4aeabed9802c7fc",
}
DEST = Path(__file__).resolve().parents[1] / "vendor" / "tinycd" / "models"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        url = f"{UPSTREAM}/{COMMIT}/models/{name}"
        with urlopen(url, timeout=60) as response:
            data = response.read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            print(f"SHA-256 mismatch for {name}: got {digest}", file=sys.stderr)
            return 1
        (DEST / name).write_bytes(data)
        print(f"ok {name}")
    print(f"TinyCD model code from upstream commit {COMMIT[:7]} written to {DEST}")
    print("Upstream terms: non-commercial and research purposes only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
