"""Choose each T policy and the primary challenger using validation data only."""
import csv
import json
from statistics import mean

from direction_protocol import CACHE, CUTOFFS, OUTPUT, ROOT, SEEDS, combine, weights
from direction_resources import wait_for_memory
from study_io import (
    MANIFEST, ORIGINAL, WORK, counts, digest, dump, images, load_tinycd_model,
    manifest, metrics, pair, probability, restore, write,
)


def load_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def probabilities(model, device, a, b, truth_shape):
    return tuple(restore(probability(model, device, x, y), truth_shape) for x, y in ((a, b), (b, a)))


def validation(family, seed):
    weight = weights(family, seed)
    names = [r["image_id"] for r in manifest("validation")]
    policies = [("forward", .5), ("reverse", .5)]
    policies.extend((mode, threshold) for mode in ("mean", "max") for threshold in CUTOFFS)
    totals = {p: dict(tp=0, fp=0, fn=0) for p in policies}
    model, device = load_tinycd_model(weight)
    for name in names:
        a, b, truth = pair("validation", name)
        a, b = images(a, b, "half")
        forward, reverse = probabilities(model, device, a, b, truth.shape)
        probs = dict(forward=forward, reverse=reverse, mean=combine(forward, reverse, "mean"),
                     max=combine(forward, reverse, "max"))
        for (mode, t), total in totals.items():
            row = counts(probs[mode] >= t, truth)
            for key in total:
                total[key] += row[key]
    return [dict(model=family, seed=seed, mode=m, cutoff=t, pairs=len(names), **metrics([total]))
            for (m, t), total in totals.items()]


def main():
    old = json.loads((ROOT / "docs/generalization-final-config.json").read_text())
    assert digest(weights("A", 42)) == old["adapted_sha256"]
    assert digest(ORIGINAL) == old["original_sha256"]
    assert all(digest(MANIFEST / f"{s}.csv") == v for s, v in old["manifests"].items())
    rows, epochs, models = [], [], []
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for family in ("A", "S"):
        for seed in SEEDS:
            weight_sha = digest(weights(family, seed))
            run = CACHE / f"{family}-{seed}"
            run.mkdir(exist_ok=True)
            checkpoint = run / "validation.json"
            if checkpoint.exists():
                record = json.loads(checkpoint.read_text())
                assert record["weights"] == weight_sha
            else:
                wait_for_memory()
                record = dict(weights=weight_sha, rows=validation(family, seed))
                dump(checkpoint, record)
            rows.extend(record["rows"])
            history = load_csv((WORK if (family, seed) == ("A", 42) else run) / "training.csv")
            assert [int(r["epoch"]) for r in history] == [1, 2, 3]
            for r in history:
                epochs.append(dict(model=family, seed=seed, epoch=r["epoch"], swaps=r.get("swaps",0),
                                   loss=r["loss"], validation_f1=r["f1"]))
            best = max(history, key=lambda r: float(r["f1"]))
            fwd = next(r for r in record["rows"] if r["mode"] == "forward")
            assert abs(float(best["f1"]) - fwd["f1"]) < 1e-12, (family, seed, best, fwd)
            models.append(dict(model=family, seed=seed, epoch=int(best["epoch"]), weights_sha256=weight_sha))
            print("Validated",family,seed, flush=True)
    write(OUTPUT / "validation.csv", rows)
    write(OUTPUT / "epochs.csv", epochs)
    families, candidates = {}, []
    for family in ("A", "S"):
        for mode, t in [("forward", .5)] + [(m, val) for m in ("mean","max") for val in CUTOFFS]:
            group = [r for r in rows if (r["model"],r["mode"],r["cutoff"]) == (family,mode,t)]
            assert {r["seed"] for r in group} == set(SEEDS) and len(group) == len(SEEDS)
            name = family if mode == "forward" else family + "+T"
            candidates.append(dict(family=name, mode=mode, cutoff=t, mean_validation_f1=mean(r["f1"] for r in group)))
        for name in (family, family + "+T"):
            # Enumeration order is the preregistered tie break.
            families[name] = max((c for c in candidates if c["family"] == name),
                                 key=lambda c: c["mean_validation_f1"])
    challenger = max(("S","A+T","S+T"), key=lambda k: families[k]["mean_validation_f1"])
    write(OUTPUT / "validation_candidates.csv", candidates)
    dump(OUTPUT / "frozen.json", dict(families=families, primary_challenger=challenger, models=models,
         manifests=old["manifests"], levir_test_manifest=old["levir_test_manifest"],
         upstream_checkpoint=old["original_sha256"], previous_config_sha256=digest(ROOT/"docs/generalization-final-config.json"),
         baseline_commit="595c3bb15f74c0c64c60b3ee0eed1b9146d126ef",
         validation_sha256=digest(OUTPUT/"validation.csv"), epoch_history_sha256=digest(OUTPUT/"epochs.csv")))
    print("Primary challenger:", challenger, flush=True)


if __name__ == "__main__":
    main()
