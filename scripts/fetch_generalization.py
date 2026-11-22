"""Fetch the pinned IGN/FOTBCD releases and verify the publishers' file digests."""
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/generalization'
BINARY = '7ebaa487d49215fdc684461ee52b41adafd83e86'
FULL = '92b677c881f0d973d76460204431c711ccbac25b'


def fetch(repo, revision, directory, suffix):
    url = f'https://huggingface.co/api/datasets/abdevd/{repo}/tree/{revision}/{directory}?limit=100'
    files = json.load(urlopen(url))
    for item in sorted(files, key=lambda x: ('train-' in x['path'], x['path'])):
        if not item['path'].endswith(suffix):
            continue
        name = Path(item['path']).name
        dest = RAW / name
        if dest.is_file() and dest.stat().st_size == item['size']:
            with dest.open('rb') as f:
                if hashlib.file_digest(f,'sha256').hexdigest() == item['lfs']['oid']:
                    continue
        tmp = dest.with_suffix('.part')
        source = f'https://huggingface.co/datasets/abdevd/{repo}/resolve/{revision}/{item["path"]}?download=true'
        subprocess.run(['curl','-fsSL','--retry','4','-C','-','-o',str(tmp),source], check=True)
        with tmp.open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest() != item['lfs']['oid']:
                raise ValueError(f'Hash mismatch: {name}')
        tmp.replace(dest)
        print(name, item['size'], flush=True)


if __name__ == '__main__':
    RAW.mkdir(exist_ok=True, parents=True)
    fetch('FOTBCD-Binary', BINARY, 'data', '.parquet')
    for split in ['val','test']:
        fetch('FOTBCD', FULL, f'data/{split}', '.tar')
