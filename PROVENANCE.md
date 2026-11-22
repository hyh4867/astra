# ASTRA data and model provenance

Checked/recorded 2026-10-31 (Asia/Seoul). Raw archives and checkpoint remain local and are excluded from Git.

## LEVIR-CD files

Official provider page: <https://justchenhao.github.io/LEVIR/>. It describes 637 image pairs and academic-only, non-commercial use; follow Google Earth terms.

Archives used locally were downloaded from the Hugging Face mirror (not the provider's Google Drive):

| File | Direct source | Download date (local file time) | Bytes | SHA-256 | Published canonical MD5 check |
|---|---|---|---:|---|---|
| `val.zip` | <https://huggingface.co/datasets/satellite-image-deep-learning/LEVIR-CD/resolve/main/val.zip> | 2026-10-31 12:03 KST | 246,152,048 | `0739bb257b0237631e58d196da6530bc82ff2eb5acb8056cd8444d2f94d058c7` | `f7b857978524f9aa8c3bf7f94e3047a4` — matches [TorchGeo's LEVIR-CD split metadata](https://docs.torchgeo.org/en/v0.8.0/_modules/torchgeo/datasets/levircd.html) |
| `test.zip` | <https://huggingface.co/datasets/satellite-image-deep-learning/LEVIR-CD/resolve/main/test.zip> | 2026-10-31 12:04 KST | 496,305,323 | `0bdf73ebd220f392fd58dea8b6cb0e369a8d8d2f59c29827d2349178e3131287` | `07d5dd89e46f5c1359e2eca746989ed9` — matches [TorchGeo's LEVIR-CD split metadata](https://docs.torchgeo.org/en/v0.8.0/_modules/torchgeo/datasets/levircd.html) |
| `train.zip` | <https://huggingface.co/datasets/satellite-image-deep-learning/LEVIR-CD/resolve/main/train.zip> | 2026-10-31 (KST) | 1,721,956,862 | `ee3a65a73d2ae88f691ff342a7124f8c8b573eb9b9ec1dd67582ebde3fa87625` | `a638e71f480628652dea78d8544307e4` — matches [TorchGeo's LEVIR-CD split metadata](https://docs.torchgeo.org/en/v0.8.0/_modules/torchgeo/datasets/levircd.html) |

The downloaded archives' MD5 values match published hashes for all three canonical splits. Extracted counts are 445/64/128 pairs. Lists of sorted names are in `data/manifests/{train,val,test}_filenames.txt`. A/B/label names match exactly within each split; all images and labels are 1024×1024. The training archive is local-only and excluded from Git.

## TinyCD checkpoint and source

Official repository: <https://github.com/AndreaCodegoni/Tiny_model_4_CD>. Checkpoint URL: <https://raw.githubusercontent.com/AndreaCodegoni/Tiny_model_4_CD/main/pretrained_models/levir_best.pth>. Downloaded 2026-10-31 12:00 KST, 1,276,499 bytes; SHA-256 `743645b87199c96bd0c3271ae5df6f0918b3f1d84618dfb9396843ca08ee4a3e`. Checkpoint is local-only and excluded from Git.

TinyCD model source files are not included in this repository: the upstream README states that the code is for non-commercial and research purposes only. `scripts/install_tinycd.py` downloads them from a pinned upstream commit and checks their SHA-256; see `vendor/tinycd/NOTICE.md`.

## Input and label audit

- `scripts/evaluate_benchmark.py:47-49` opens A/B with Pillow `.convert("RGB")`; this yields RGB arrays. TinyCD gets those RGB arrays at lines 173-175, converts HWC to CHW and `[0,255]` to `[0,1]`, then applies ImageNet RGB normalization. It does not use `cv2.imread`/BGR.
- OpenCV uses `cv2.COLOR_RGB2GRAY` on the same RGB arrays (`scripts/evaluate_benchmark.py:54` for cached diff; `:136-138` for timed basic inference), not BGR grayscale.
- Labels are loaded as grayscale and binarized with `label > 127` (`scripts/evaluate_benchmark.py:49-53`). Across all val/test masks the unique values are `[0, 156, 254, 255]`; no other values were found. Thus the previous `>0` and current `>127` produce identical masks in this data, though 156/254 are noncanonical intermediate values.
- Project venv architecture: `arm64` (`platform.machine()`); not x86_64/Rosetta.
- Existing-run accounting check: each split has exactly `pair_count × 1024 × 1024` pixels. Validation has 67,108,864 pixels; test has 134,217,728. Derived TN: validation OpenCV 59,247,159; validation TinyCD 64,073,392; test OpenCV 117,038,087; test TinyCD 126,895,451. The evaluator now asserts `TP+FP+FN+TN == pair_count × 1024 × 1024`.

Code paths are noted above. The evaluator checks TP+FP+FN+TN at both per-image and aggregate levels for every method.

Latest-run TN values: validation OpenCV basic 59,247,159; OpenCV improved 48,100,645; TinyCD 64,073,392. Test OpenCV basic 117,038,087; OpenCV improved 93,734,770; OpenCV oracle 114,639,856; TinyCD 126,895,451. Each sums with TP/FP/FN to the corresponding total pixels in the CSV.

## Advanced fine-tuning provenance

The upstream `training.py` and `dataset/dataset.py` were inspected from <https://github.com/AndreaCodegoni/Tiny_model_4_CD>. Upstream uses BCELoss, AdamW (LEVIR lr 0.00356799066427741, weight decay 0.009449677083344786), CosineAnnealingLR T_max=100, 100 epochs, batch size 8. Augmentations: paired horizontal/vertical flip (`Flip`, p=0.5), paired rotation within ±5° (p=0.5), then independent per-image brightness ±0.2 and contrast ±0.2 (p=0.5), and independent Gaussian blur kernel 3 or 5 (p=0.5); there is no hue/saturation jitter. The model instantiated from the vendored upstream architecture has 285,128 parameters. Advanced run settings and selected checkpoints/results will be recorded separately; checkpoints stay local-only.

## FOTBCD external evaluation

Abdelrrahman Moubane, *FOTBCD: A Large-Scale Building Change Detection Benchmark from French Orthophotos and Topographic Data* (2026). The original imagery and topographic data are IGN BD ORTHO and BD TOPO under Licence Ouverte 2.0. The author distributes both FOTBCD and FOTBCD-Binary under CC BY 4.0. This study resizes imagery for selected methods and rasterizes NEW/DEMOLISHED polygons for explanatory recall. [Paper](https://arxiv.org/abs/2601.22596) · [Publisher and attribution](https://github.com/abdelpy/FOTBCD-datasets/tree/26daefbaec07b5bdd49ee65a84f4113c1913b108) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

From the project directory, run `python scripts/fetch_generalization.py` to download and verify the pinned Binary Parquet shards and the full edition's validation and test TARs. `python scripts/freeze_generalization.py` reads only Binary metadata and recreates the selection manifests. Numeric image IDs are split-local. Cross-edition joins use split, department, CRS, acquisition years and footprint coordinates rounded to one millimetre. The input archives, polygons, images, prediction masks and adapted weights stay in the ignored local data directory. [Binary revision](https://huggingface.co/datasets/abdevd/FOTBCD-Binary/tree/7ebaa487d49215fdc684461ee52b41adafd83e86) · [Full revision](https://huggingface.co/datasets/abdevd/FOTBCD/tree/92b677c881f0d973d76460204431c711ccbac25b)

The split manifests are attributed adaptations of Moubane’s CC BY 4.0 release. They record split-local IDs, shard names and row positions. The test list preserves the complete official split. Training rows are a hash-selected subset. To regenerate, install PyArrow, run the two scripts above, then compare the SHA-256 of each CSV with [the split audit](data/manifests/fotbcd/audit.json).
