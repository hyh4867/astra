"""Safely extract the verified LEVIR-CD train archive into Git-ignored storage."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/raw/train.zip"
DEST = ROOT / "data/benchmark/train"
MANIFEST = ROOT / "data/manifests/train_filenames.txt"


def main() -> None:
    if not ARCHIVE.is_file():
        raise FileNotFoundError(ARCHIVE)
    DEST.mkdir(parents=True, exist_ok=True)
    with ZipFile(ARCHIVE) as archive:
        for member in archive.infolist():
            rel = PurePosixPath(member.filename)
            if rel.is_absolute() or ".." in rel.parts or not rel.parts:
                raise ValueError(f"unsafe archive member: {member.filename}")
            if rel.parts[0] not in {"A", "B", "label"}:
                raise ValueError(f"unexpected archive folder: {member.filename}")
            if member.is_dir():
                continue
            if len(rel.parts) != 2:
                raise ValueError(f"unexpected nested path: {member.filename}")
            if rel.suffix.lower() != ".png":
                raise ValueError(f"unexpected file type: {member.filename}")
            target = DEST.joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size != member.file_size:
                with archive.open(member) as src, target.open("wb") as dst:
                    while block := src.read(1024 * 1024):
                        dst.write(block)

    sets = [{p.name for p in (DEST / folder).glob("*.png")} for folder in ("A", "B", "label")]
    if not sets[0] or sets[0] != sets[1] or sets[0] != sets[2]:
        raise RuntimeError(f"train split file lists differ: {[len(s) for s in sets]}")
    names = sorted(sets[0])
    if len(names) != 445:
        raise RuntimeError(f"expected 445 pairs, found {len(names)}")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text("".join(f"{name}\n" for name in names))
    print(f"extracted and verified {len(names)} train pairs under {DEST}")


if __name__ == "__main__":
    main()
