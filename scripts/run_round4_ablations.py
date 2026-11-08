"""Train the two pre-specified Round 4 ablations, then evaluate them locally."""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor/tinycd"))

from evaluate_benchmark import Sample, load_samples, tinycd_predict
from finetune_tinycd_robust import BASE, PATCH, TRAIN, VAL, normalize_batch, paired_geometry, upstream_photometric
from models.change_classifier import ChangeClassifier
from finetune_tinycd_v2 import freeze_bn

OUT = ROOT / "outputs/diagnosis_round"
LOG = ROOT / "outputs/finetune"
CROP = 8
SEED = 42
EPOCHS = 5
WEIGHT_DECAY = 0.009449677083344786
PRETRAINED = ROOT / "models/tinycd_levir_best.pth"
CHECKPOINTS = {
    "ROBUST-v1": ROOT / "models/finetuned/tinycd_robust_best.pth",
    "ABL-A (v1 B-shift; BN frozen)": ROOT / "models/finetuned/tinycd_round4_abla_shift_seed42_best.pth",
    "ABL-B (BN train; no shift)": ROOT / "models/finetuned/tinycd_round4_abla_bn_seed42_best.pth",
    "ROBUST-v2 seed 42": ROOT / "models/finetuned/tinycd_robust_v2_seed42_best.pth",
}


def training_item(name: str, index: int, epoch: int, shift: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED + epoch * 1_000_003 + index)
    a = np.asarray(Image.open(TRAIN / "A" / name).convert("RGB"))
    b = np.asarray(Image.open(TRAIN / "B" / name).convert("RGB"))
    label = (np.asarray(Image.open(TRAIN / "label" / name).convert("L")) > 127).astype(np.uint8)
    a, b, label = paired_geometry(a, b, label, rng)
    a, b = upstream_photometric(a, rng), upstream_photometric(b, rng)
    y = int(rng.integers(0, a.shape[0] - PATCH + 1))
    x = int(rng.integers(0, a.shape[1] - PATCH + 1))
    a, b, label = a[y:y + PATCH, x:x + PATCH], b[y:y + PATCH, x:x + PATCH], label[y:y + PATCH, x:x + PATCH]
    for index_image in (0, 1):
        if rng.random() < 0.5:
            sigma = float(rng.uniform(0.0, 2.5))
            if sigma > 0:
                if index_image == 0:
                    a = cv2.GaussianBlur(a, (0, 0), sigmaX=sigma)
                else:
                    b = cv2.GaussianBlur(b, (0, 0), sigmaX=sigma)
    if shift and rng.random() < 0.5:
        dx, dy = int(rng.integers(-3, 4)), int(rng.integers(-3, 4))
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        # Deliberately reproduce the v1 bug for this diagnostic ablation:
        # shift B while leaving the undirected label unchanged.
        b = cv2.warpAffine(b, matrix, (PATCH, PATCH), borderMode=cv2.BORDER_REFLECT_101)
    if rng.random() < 0.5:
        a, b = b, a
    return a, b, label.astype(np.float32)


def counts(pred: np.ndarray, truth: np.ndarray) -> dict[str, int | float]:
    p, y = pred.astype(bool), truth.astype(bool)
    tp, fp, fn = int(np.count_nonzero(p & y)), int(np.count_nonzero(p & ~y)), int(np.count_nonzero(~p & y))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def evaluate_val(model: torch.nn.Module, blur_sigma: float = 0.0) -> float:
    model.eval()
    total = {"tp": 0, "fp": 0, "fn": 0}
    names = sorted(path.name for path in (VAL / "A").glob("*.png"))
    with torch.inference_mode():
        for name in names:
            a = np.asarray(Image.open(VAL / "A" / name).convert("RGB"))
            b = np.asarray(Image.open(VAL / "B" / name).convert("RGB"))
            y = np.asarray(Image.open(VAL / "label" / name).convert("L")) > 127
            if blur_sigma:
                b = cv2.GaussianBlur(b, (0, 0), sigmaX=blur_sigma)
            sample = Sample(name, a, b, y, np.zeros_like(y, dtype=np.uint8))
            result = counts(tinycd_predict(model, sample, torch.device("cpu")), y)
            for key in total:
                total[key] += int(result[key])
    return float(counts_from_totals(total)["f1"])


def counts_from_totals(total: dict[str, int]) -> dict[str, float]:
    p = total["tp"] / (total["tp"] + total["fp"]) if total["tp"] + total["fp"] else 0.0
    r = total["tp"] / (total["tp"] + total["fn"]) if total["tp"] + total["fn"] else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def train_ablation(name: str, checkpoint: Path, shift: bool, freeze_batchnorm: bool) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    names = sorted(path.name for path in (TRAIN / "A").glob("*.png"))
    if not names:
        raise FileNotFoundError(f"LEVir train data not available at {TRAIN / 'A'}")
    model = ChangeClassifier(pretrained=False)
    model.load_state_dict(torch.load(PRETRAINED, map_location="cpu", weights_only=True), strict=True)
    model.to("cpu")
    if freeze_batchnorm:
        freeze_bn(model)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = torch.nn.BCELoss()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    epoch_dir = checkpoint.parent / f"{checkpoint.stem}_epochs"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG / f"round4_{checkpoint.stem}_epochs.csv"
    LOG.mkdir(parents=True, exist_ok=True)
    rows = []
    best = -1.0
    for epoch in range(EPOCHS):
        started = time.perf_counter()
        model.train()
        if freeze_batchnorm:
            freeze_bn(model)
        order = np.random.default_rng(SEED + epoch).permutation(len(names))
        losses = []
        for step, raw_index in enumerate(order, 1):
            a, b, label = training_item(names[int(raw_index)], int(raw_index), epoch, shift)
            xa = normalize_batch([a])
            xb = normalize_batch([b])
            y = torch.from_numpy(label[None, ...])
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xa, xb).squeeze(1)
            loss = loss_fn(prediction, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            if step % 250 == 0 or step == len(order):
                print(f"{name} epoch={epoch + 1}/{EPOCHS} train={step}/{len(order)}", flush=True)
        clean_val = evaluate_val(model)
        blur_val = evaluate_val(model, blur_sigma=2.0)
        elapsed = time.perf_counter() - started
        row = {"ablation": name, "epoch": epoch + 1, "train_loss": float(np.mean(losses)),
               "clean_val_f1": clean_val, "B_blur_sigma2_val_f1": blur_val,
               "epoch_seconds": elapsed, "seed": SEED, "device": "cpu", "batch_size": 1,
               "batchnorm_frozen": freeze_batchnorm, "B_shift_enabled": shift}
        rows.append(row)
        torch.save(model.state_dict(), epoch_dir / f"epoch_{epoch + 1:02d}.pth")
        if clean_val > best:
            best = clean_val
            torch.save(model.state_dict(), checkpoint)
        print(json.dumps(row), flush=True)
        scheduler.step()
    with log_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"selected {name}: cleanVAL={best:.6f}; checkpoint={checkpoint.relative_to(ROOT)}", flush=True)


def predict_cropped(model: torch.nn.Module, sample: Sample, before: np.ndarray, after: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    updated = Sample(sample.name, before, after, sample.truth, sample.gray_diff)
    prediction = tinycd_predict(model, updated, torch.device("cpu"))
    return prediction[CROP:-CROP, CROP:-CROP], sample.truth[CROP:-CROP, CROP:-CROP]


def evaluate_test() -> list[dict]:
    names = list(CHECKPOINTS)
    paths = [CHECKPOINTS[name] for name in names]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("At least one requested ablation checkpoint is missing")
    samples = list(load_samples("test"))
    subset_names = (OUT / "subset_seed42_sorted.txt").read_text().splitlines()
    by_name = {sample.name: sample for sample in samples}
    subset = [by_name[name] for name in subset_names]
    rows = []
    for name, checkpoint in zip(names, paths):
        model = ChangeClassifier(pretrained=False)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        model.eval()
        clean_total = {"tp": 0, "fp": 0, "fn": 0}
        started = time.perf_counter()
        with torch.inference_mode():
            for sample in samples:
                result = counts(tinycd_predict(model, sample, torch.device("cpu")), sample.truth)
                for key in clean_total:
                    clean_total[key] += int(result[key])
        clean_runtime = time.perf_counter() - started
        clean = counts_from_totals(clean_total)
        row = {"model": name, "clean_test_precision": clean["precision"],
               "clean_test_recall": clean["recall"], "clean_test_f1": clean["f1"],
               "clean_test_runtime_s": clean_runtime}
        for condition in ("B_blur_sigma2", "A_blur_sigma2", "AB_blur_sigma2", "swap_BA"):
            total = {"tp": 0, "fp": 0, "fn": 0}
            condition_started = time.perf_counter()
            with torch.inference_mode():
                for sample in subset:
                    a, b = sample.before, sample.after
                    if condition == "B_blur_sigma2":
                        b = cv2.GaussianBlur(b, (0, 0), sigmaX=2.0)
                    elif condition == "A_blur_sigma2":
                        a = cv2.GaussianBlur(a, (0, 0), sigmaX=2.0)
                    elif condition == "AB_blur_sigma2":
                        a = cv2.GaussianBlur(a, (0, 0), sigmaX=2.0)
                        b = cv2.GaussianBlur(b, (0, 0), sigmaX=2.0)
                    elif condition == "swap_BA":
                        a, b = b, a
                    prediction, truth = predict_cropped(model, sample, a, b)
                    result = counts(prediction, truth)
                    for key in total:
                        total[key] += int(result[key])
            condition_metrics = counts_from_totals(total)
            row[f"{condition}_precision"] = condition_metrics["precision"]
            row[f"{condition}_recall"] = condition_metrics["recall"]
            row[f"{condition}_f1"] = condition_metrics["f1"]
            row[f"{condition}_fp"] = total["fp"]
            row[f"{condition}_runtime_s"] = time.perf_counter() - condition_started
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "round4_ablation_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (OUT / "round4_ablation_metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def main() -> None:
    if not PRETRAINED.is_file():
        raise FileNotFoundError(PRETRAINED)
    train_ablation("ABL-A (v1 B-shift; BN frozen)", CHECKPOINTS["ABL-A (v1 B-shift; BN frozen)"], shift=True, freeze_batchnorm=True)
    train_ablation("ABL-B (BN train; no shift)", CHECKPOINTS["ABL-B (BN train; no shift)"], shift=False, freeze_batchnorm=False)
    evaluate_test()


if __name__ == "__main__":
    main()
