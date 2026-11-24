"""Repeat the original adapted training, with an independent optional time-swap stream."""
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F

from direction_protocol import CACHE, SEEDS, temporal_pair, weights
from direction_resources import wait_for_memory
from study_io import (
    counts, digest, images, load_tinycd_model, manifest, metrics, pair, probability, restore, write,
)
from study_train import train_pair


def save(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def train(family, seed):
    if (family, seed) == ("A", 42):
        raise ValueError("A42 is the previously frozen adapted checkpoint")
    out = CACHE / f"{family}-{seed}"
    out.mkdir(exist_ok=True, parents=True)
    state_path = out / "epoch.pth"
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model, device = load_tinycd_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
    history = []
    best = None
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        assert (state["family"], state["seed"]) == (family, seed)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, best = state["history"], state["best"]
    names = [r["image_id"] for r in manifest("train")]
    validation = [r["image_id"] for r in manifest("validation")]
    assert len(names) == 1024 and len(validation) == 984
    for epoch in range(len(history) + 1, 4):
        torch.manual_seed(seed + epoch)
        rng = random.Random(seed + epoch)
        swap_rng = random.Random(f"temporal-swap/{seed + epoch}")
        order = rng.sample(names, len(names))
        model.train()
        losses, swaps = [], 0
        for idx in range(0, len(order), 2):
            group = []
            for name in order[idx:idx + 2]:
                sample = train_pair(name, rng.random() < 0.5)
                swap = family == "S" and swap_rng.random() < 0.5
                swaps += int(swap)
                group.append(temporal_pair(*sample, swap))
            a, b, y = [torch.stack(t).to(device) for t in zip(*group)]
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy(model(a, b)[:, 0, :205, :205], y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if idx % 256 == 0:
                print(f"{family}{seed} epoch {epoch}: {idx}/{len(names)}", flush=True)
        model.eval()
        rows = []
        for name in validation:
            a, b, g = pair("validation", name)
            a, b = images(a, b, "half")
            rows.append(counts(restore(probability(model, device, a, b), g.shape) >= .5, g))
        result = dict(epoch=epoch, swaps=swaps, loss=float(np.mean(losses)), **metrics(rows))
        if not history or result["f1"] > max(r["f1"] for r in history):
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(result)
        save(dict(family=family, seed=seed, model=model.state_dict(), optimizer=optimizer.state_dict(),
                  history=history, best=best), state_path)
        print(result, flush=True)
    save(best, weights(family, seed))
    write(out / "training.csv", history)
    print(f"{family}{seed} best weights {digest(weights(family, seed))}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=["A", "S"])
    parser.add_argument("seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    wait_for_memory(training=True)
    train(args.family, args.seed)
