# ASTRA

A local research app that maps new buildings between two satellite images with a pretrained TinyCD model, scored against fixed baselines on a held-out benchmark and checked under misalignment and blur.

Built with AI coding agents (Claude, Codex); tests and verification by me.

![Four panels from a public-domain NAIP image pair near Celina, Texas: the 2020 image, the 2022 image, the TinyCD change probability and the candidate overlay](docs/naip_qualitative.png)

Qualitative side demo on public-domain USDA NAIP imagery (Celina, Texas; 2020-11-02 and 2022-07-04; 0.6 m). There are no ground-truth labels for this pair, so it is not a score.

## Results

Held-out LEVIR-CD test set, 128 image pairs; the model was not tuned on these pairs.

| Check | Result | Source |
|---|---|---|
| Pretrained TinyCD, pixel level | Precision 0.9266, recall 0.8953, F1 0.9107, IoU 0.8360 | `outputs/benchmark_test.csv` |
| OpenCV baselines, pixel level | F1 0.1306 (basic) and 0.1003 (improved) | `outputs/benchmark_test.csv` |
| TinyCD, object level | Completeness 0.957 against 0.071 for a chance baseline (predictions paired with a different image, seed 42), giving an object skill of 0.953 | `outputs/object_null_completeness_seed42.csv` (summed over size bins) |
| Input shifted by 5 px in x and y, fixed 32-pair subset | F1 fell from 0.9211 (control) to 0.7229 | `outputs/robustness_test_seed42.csv` |
| False alarms on LEVIR, objects ≥ 5 m² | 7.48 false-alarm objects per km² | `outputs/isolated_object_summary.csv` |
| Same 128 test pairs on Apple MPS and on CPU | Identical true-positive, false-positive and false-negative pixel counts | `outputs/tinycd_test_mps_comparison.csv` |
| Model size | The pretrained TinyCD model has 285,128 parameters (the model's own count, not parameters trained for ASTRA) | `PROVENANCE.md` |
| Test suite | 12 tests pass with TinyCD installed; 11 pass and the TinyCD smoke test is skipped with the base install | `tests/` |

## How it was verified

- **Held-out benchmark.** Scores come from the official LEVIR-CD test split (128 pairs) with a fixed threshold. The OpenCV baselines (absolute difference with a fixed or Otsu threshold) are scored the same way.
- **A chance baseline for objects.** Object completeness is compared with a deranged null: each ground-truth image is paired with predictions from a different test image (seed 42). Object skill is (real − chance) / (1 − chance).
- **Robustness under stated conditions.** Shifts of 1, 2, 3 and 5 px (equal in x and y), gain, gamma, Gaussian blur and noise were applied to a fixed 32-pair subset (seed 42, 8 px border cropped). TinyCD's F1 was 0.9026, 0.8606, 0.8128 and 0.7229 for the four shifts, against 0.9211 without perturbation.
- **Device check.** The full test split was run on MPS and on CPU, and the confusion counts were required to match exactly.
- **Tests** cover the OpenCV detector, padding and tiling, the GSD-scaled minimum object area, input-quality warnings, GeoTIFF reading and alignment, and a Streamlit smoke test for each method on a synthetic pair.

## Limits

- Benchmark scores describe the stated LEVIR-CD test and do not establish performance beyond it. The model is appearance-oriented; detecting disappearance is not validated.
- Robustness results are limited to the fixed 32-pair subset and the stated perturbations. They say nothing about other sensors.
- The NAIP figure is a qualitative demonstration with no ground truth, not a scored evaluation.
- The 7.48 false-alarm objects per km² is a LEVIR benchmark yardstick, not a false-alarm rate measured on NAIP.
- TinyCD's code and checkpoint are for non-commercial and research use only (see below), which limits how the full app can be used.

## Run it

Python 3.12.

```sh
git clone https://github.com/hyh4867/astra.git
cd astra
python3.12 -m venv .venv && . .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

With the base install, 11 tests pass and the TinyCD smoke test is skipped, and the app's OpenCV methods work; nothing is downloaded. The TinyCD method needs PyTorch, the TinyCD code and its checkpoint; after these steps all 12 tests pass:

```sh
python -m pip install -r requirements-ml.txt
python scripts/install_tinycd.py          # TinyCD code from a pinned upstream commit, SHA-256 checked
python scripts/download_tinycd_weights.py # pretrained LEVIR-CD checkpoint -> models/ (not committed)
python -m pytest -q
```

The fine-tuned ROBUST-v2 checkpoint mentioned in the app is not published (it derives from the non-commercial TinyCD weights and LEVIR-CD), so the app offers only the original model.

To reproduce the benchmark files in `outputs/`, download LEVIR-CD with `python scripts/download_levir_benchmark.py` (val and test archives from the Hugging Face mirror; about 0.7 GB) and run `python scripts/evaluate_benchmark.py`. The robustness and object-level files come from `scripts/scientific_evaluation.py` and `scripts/final_analysis_additions.py`. These steps need the data and were not re-run for this repository. Archive hashes and split counts are recorded in `PROVENANCE.md`; the file lists for each split are in `data/manifests/`.

## Data, citation and licenses

The datasets, the TinyCD code and the checkpoint are not redistributed here.

| Item | Source | Terms |
|---|---|---|
| LEVIR-CD (637 image pairs) | Provider page <https://justchenhao.github.io/LEVIR/>; mirror used: Hugging Face `satellite-image-deep-learning/LEVIR-CD` | Academic, non-commercial use only; follow Google Earth terms. Only file names are included here (`data/manifests/`). |
| TinyCD code and LEVIR-CD checkpoint | <https://github.com/AndreaCodegoni/Tiny_model_4_CD> | Non-commercial and research purposes only; no open-source license. Downloaded by `scripts/install_tinycd.py` and `scripts/download_tinycd_weights.py`. See `vendor/tinycd/NOTICE.md`. |
| USDA NAIP imagery (demo figure) | Microsoft Planetary Computer STAC, collection `naip` (`scripts/acquire_naip_demo.py`) | Public domain. |

- Chen & Shi, A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection, *Remote Sensing* 12(10), 1662 (2020), [doi:10.3390/rs12101662](https://doi.org/10.3390/rs12101662).
- Codegoni, Lombardi & Ferrari, TINYCD: A (Not So) Deep Learning Model For Change Detection, [arXiv:2207.13159](https://arxiv.org/abs/2207.13159) (2022).
- Dependencies: Streamlit (Apache-2.0), OpenCV headless (Apache-2.0), NumPy (BSD-3-Clause), Pillow (MIT-CMU), pytest (MIT), rasterio (BSD-3-Clause), pystac-client (Apache-2.0), planetary-computer (MIT), folium and streamlit-folium (MIT); optional PyTorch (BSD-style) and torchvision (BSD-3-Clause).
- Code: MIT License, see `LICENSE`. The MIT License covers ASTRA's own code only, not TinyCD.
