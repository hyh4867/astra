"""Clean-validation-selected TinyCD fine-tuning for ASTRA's robustness study.

Train data are sampled as one random 256px crop per 1024px LEVIR pair per
epoch. No test image is opened by this script. Checkpoints/logs are local-only.
"""
from __future__ import annotations

import argparse
import csv
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
sys.path.insert(0, str(ROOT / "vendor" / "tinycd"))
from models.change_classifier import ChangeClassifier
from inference import tinycd_probability

TRAIN = ROOT / "data/benchmark/train"
VAL = ROOT / "data/benchmark/val"
BASE = ROOT / "models/tinycd_levir_best.pth"
OUT = ROOT / "models/finetuned"
LOG = ROOT / "outputs/finetune"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
PATCH = 256


def paired_geometry(a: np.ndarray, b: np.ndarray, mask: np.ndarray, rng):
    if rng.random() < 0.5:
        code = int(rng.integers(0, 3))
        a, b, mask = cv2.flip(a, code), cv2.flip(b, code), cv2.flip(mask, code)
    if rng.random() < 0.5:
        h, w = mask.shape
        angle = float(rng.uniform(-5, 5))
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        a = cv2.warpAffine(a, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        b = cv2.warpAffine(b, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return a, b, mask


def upstream_photometric(image: np.ndarray, rng):
    result = image
    if rng.random() < 0.5:
        brightness = float(rng.uniform(-0.2, 0.2)) * 255
        contrast = float(rng.uniform(0.8, 1.2))
        result = np.clip((result.astype(np.float32) - 127.5) * contrast + 127.5 + brightness, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        result = cv2.GaussianBlur(result, (int(rng.choice([3, 5])),) * 2, 0)
    return result


def normalize_batch(images):
    arr = np.stack(images).astype(np.float32) / 255.0
    arr = (arr - MEAN[None, None, None, :]) / STD[None, None, None, :]
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def make_train_item(name: str, index: int, epoch: int, robust: bool):
    rng = np.random.default_rng(42 + epoch * 1_000_003 + index)
    a = np.asarray(Image.open(TRAIN / "A" / name).convert("RGB"))
    b = np.asarray(Image.open(TRAIN / "B" / name).convert("RGB"))
    mask = (np.asarray(Image.open(TRAIN / "label" / name).convert("L")) > 127).astype(np.uint8)
    a, b, mask = paired_geometry(a, b, mask, rng)
    a, b = upstream_photometric(a, rng), upstream_photometric(b, rng)
    y = int(rng.integers(0, a.shape[0] - PATCH + 1))
    x = int(rng.integers(0, a.shape[1] - PATCH + 1))
    a, b, mask = a[y:y+PATCH, x:x+PATCH], b[y:y+PATCH, x:x+PATCH], mask[y:y+PATCH, x:x+PATCH]
    if robust:
        if rng.random() < 0.5:
            sigma = float(rng.uniform(0.0, 2.5))
            if sigma > 0:
                a = cv2.GaussianBlur(a, (0, 0), sigmaX=sigma)
        if rng.random() < 0.5:
            sigma = float(rng.uniform(0.0, 2.5))
            if sigma > 0:
                b = cv2.GaussianBlur(b, (0, 0), sigmaX=sigma)
        if rng.random() < 0.5:
            dx, dy = int(rng.integers(-3, 4)), int(rng.integers(-3, 4))
            matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            b = cv2.warpAffine(b, matrix, (PATCH, PATCH), borderMode=cv2.BORDER_REFLECT_101)
        if rng.random() < 0.5:
            a, b = b, a
    return a, b, mask.astype(np.float32)


def metric_counts(tp: int, fp: int, fn: int):
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def evaluate_val(model, device):
    model.eval()
    tp = fp = fn = 0
    names = sorted(p.name for p in (VAL / "A").glob("*.png"))
    with torch.inference_mode():
        for name in names:
            a = np.asarray(Image.open(VAL / "A" / name).convert("RGB"))
            b = np.asarray(Image.open(VAL / "B" / name).convert("RGB"))
            truth = np.asarray(Image.open(VAL / "label" / name).convert("L")) > 127
            pred = tinycd_probability(model, a, b, device) >= 0.5
            tp += int(np.count_nonzero(pred & truth))
            fp += int(np.count_nonzero(pred & ~truth))
            fn += int(np.count_nonzero(~pred & truth))
    return metric_counts(tp, fp, fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["control", "robust"], required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--stop-after", type=int, default=0,
                        help="optional early stop while preserving the --epochs cosine-schedule horizon")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit-samples", type=int, default=0, help="debug timing only; 0 uses all 445 pairs")
    parser.add_argument("--skip-validation", action="store_true", help="for timing probes only; never use to select a checkpoint")
    args = parser.parse_args()
    run_epochs = min(args.epochs, args.stop_after) if args.stop_after > 0 else args.epochs
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; refusing to silently train on CPU")
    device = torch.device("mps")
    names = sorted(p.name for p in (TRAIN / "A").glob("*.png"))
    if len(names) != 445:
        raise RuntimeError(f"expected 445 train pairs, found {len(names)}")
    model = ChangeClassifier(pretrained=False)
    state = torch.load(BASE, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).train()
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.009449677083344786)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    criterion = torch.nn.BCELoss()
    OUT.mkdir(parents=True, exist_ok=True); LOG.mkdir(parents=True, exist_ok=True)
    ckpt = OUT / f"tinycd_{args.mode}_best.pth"
    ready_marker = OUT / "tinycd_robust_ready.json"
    if args.mode == "robust":
        ready_marker.unlink(missing_ok=True)
    log_path = LOG / f"{args.mode}_epochs.csv"
    best_f1 = -1.0
    if log_path.exists():
        log_path.unlink()
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_precision", "val_recall", "val_f1", "val_iou", "epoch_seconds", "parameters", "device"])
        writer.writeheader()
        f.flush()
        for epoch in range(run_epochs):
            started = time.perf_counter()
            model.train()
            order = np.random.default_rng(args.seed + epoch).permutation(len(names))
            if args.limit_samples:
                order = order[:args.limit_samples]
            tp = fp = fn = 0
            loss_sum = 0.0
            for start in range(0, len(order), args.batch_size):
                chunk = order[start:start + args.batch_size]
                items = [make_train_item(names[int(idx)], int(idx), epoch, args.mode == "robust") for idx in chunk]
                xa = normalize_batch([item[0] for item in items]).to(device)
                xb = normalize_batch([item[1] for item in items]).to(device)
                target = torch.from_numpy(np.stack([item[2] for item in items])).to(device).float()
                optimizer.zero_grad(set_to_none=True)
                probabilities = model(xa, xb).squeeze(1)
                loss = criterion(probabilities, target)
                loss.backward(); optimizer.step()
                loss_sum += float(loss.detach().cpu())
                pred = (probabilities.detach() >= 0.5)
                truth = target.bool()
                tp += int((pred & truth).sum().cpu())
                fp += int((pred & ~truth).sum().cpu())
                fn += int((~pred & truth).sum().cpu())
                if (start // args.batch_size + 1) % 4 == 0 or start + args.batch_size >= len(order):
                    print(f"{args.mode} epoch {epoch+1}: {min(start + args.batch_size, len(order))}/{len(order)} train samples", flush=True)
            train = metric_counts(tp, fp, fn)
            val = evaluate_val(model, device) if not args.skip_validation else {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "iou": float("nan")}
            elapsed = time.perf_counter() - started
            row = {"epoch": epoch + 1, "train_loss": loss_sum / max(1, int(np.ceil(len(names) / args.batch_size))),
                   "val_precision": val["precision"], "val_recall": val["recall"], "val_f1": val["f1"],
                   "val_iou": val["iou"], "epoch_seconds": elapsed, "parameters": parameter_count, "device": "mps"}
            writer.writerow(row); f.flush()
            print(f"{args.mode} epoch {epoch+1}/{run_epochs} (cosine horizon {args.epochs}) train_loss={row['train_loss']:.5f} val_P/R/F1/IoU={val['precision']:.4f}/{val['recall']:.4f}/{val['f1']:.4f}/{val['iou']:.4f} time={elapsed:.1f}s", flush=True)
            if not args.skip_validation and val["f1"] > best_f1:
                best_f1 = val["f1"]
                torch.save(model.state_dict(), ckpt)
                print(f"saved best clean-val checkpoint: {ckpt}", flush=True)
            scheduler.step()
    print(f"epochs_run={run_epochs}/{args.epochs} parameters={parameter_count} best_val_f1={best_f1:.6f} checkpoint={ckpt}", flush=True)
    if args.mode == "robust" and not args.skip_validation and best_f1 >= 0 and ckpt.is_file():
        ready_marker.write_text(
            f'{{"epochs_run":{run_epochs},"cosine_horizon":{args.epochs},"best_val_f1":{best_f1:.8f},"seed":{args.seed}}}\n'
        )
        print(f"ROBUST checkpoint ready for app: {ready_marker}", flush=True)


if __name__ == "__main__":
    main()
