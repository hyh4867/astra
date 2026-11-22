"""Record the selected settings and exact inputs before the final run."""
from importlib.metadata import version
import platform
import csv
import json
from study_io import (
    MANIFEST, ORIGINAL, ROOT, WORK, digest, dump,
)
from fetch_generalization import BINARY,FULL


def config():
    return dict(binary_revision=BINARY,polygon_revision=FULL,
        manifests={s:digest(MANIFEST/(s+'.csv')) for s in ['train','validation','test']},
        original_sha256=digest(ORIGINAL),adapted_sha256=digest(WORK/'adapted.pth'),
        levir_test_manifest=digest(ROOT/'data/manifests/test_filenames.txt'),
        registration=json.loads((WORK/'registration_choice.json').read_text()),
        methods=dict(original=dict(grid='native',cutoff=.5,reg='off'),
                     calibrated=json.loads((WORK/'calibration_choice.json').read_text()),
                     adapted=dict(grid='half',cutoff=.5,reg='off')),
        adapted_epoch=max(csv.DictReader((WORK/'training.csv').open()),key=lambda r:float(r['f1']))['epoch'],
        normalization='ImageNet RGB; chronological pair; tiled 256',
        geometry='area 512 to 205, reflect-101 pad; bilinear probability to 512; unchanged GT',
        scoring='8-connected objects, .30 coverage, .10 false-alarm purity, seed42 derangement',
        python=platform.python_version(),packages={k:version(k) for k in ['torch','torchvision','numpy','pyarrow','opencv-python-headless','rasterio','Pillow']})


if __name__=='__main__':
    dump(ROOT/'docs/generalization-final-config.json',config())
