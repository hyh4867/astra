"""One hash-bound pass per test pair and weight file, shared by two frozen families."""
import json
import subprocess

from direction_protocol import CACHE, OUTPUT, ROOT, SEEDS, combine, weights
from direction_resources import wait_for_memory
from direction_select import probabilities
from evaluate_benchmark import load_samples
from study_direction import class_mask, match
from study_io import MANIFEST, counts, digest, dump, images, load_tinycd_model, manifest, pair, write


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def evaluate_rows(dataset, family, seed, config, stamp, names, polygons):
    dest = CACHE / "locked_test" / dataset / f"{family}-{seed}"
    dest.mkdir(parents=True, exist_ok=True)
    pending = [n for n in names if not (dest / f"{n}.json").exists()]
    if pending:
        wait_for_memory()
        model, device = load_tinycd_model(weights(family, seed))
        if dataset == "fotbcd":
            def samples():
                for name in pending:
                    a, b, truth = pair("test", name)
                    a, b = images(a, b, "half")
                    yield name, a, b, truth
        else:
            def samples():
                for sample in load_samples("test", pending):
                    yield sample.name, sample.before, sample.after, sample.truth
        settings = config["families"][family + "+T"]
        for idx, (name, a, b, truth) in enumerate(samples()):
            forward, reverse = probabilities(model, device, a, b, truth.shape)
            ensemble = combine(forward, reverse, settings["mode"])
            targets = {}
            if dataset == "fotbcd":
                targets = {key: class_mask(polygons[name], category)
                           for key, category in (("new",3),("disappeared",2))}
            rows = []
            for method, mask in ((family, forward >= .5), (family + "+T", ensemble >= settings["cutoff"])):
                row = dict(family=method, seed=seed, name=name, **counts(mask,truth))
                for key, gt in targets.items():
                    row[key+"_hits"] = int((gt & mask).sum())
                    row[key+"_pixels"] = int(gt.sum())
                rows.append(row)
            dump(dest / f"{name}.json", dict(stamp=stamp, rows=rows))
            if idx % 200 == 0:
                print(dataset,family,seed,idx,"/",len(pending), flush=True)
    result = []
    for name in names:
        saved = json.loads((dest / f"{name}.json").read_text())
        assert saved["stamp"] == stamp and [r["family"] for r in saved["rows"]] == [family,family+"+T"]
        assert all(r["name"] == name and r["seed"] == seed for r in saved["rows"])
        result.extend(saved["rows"])
    return result


def main():
    wait_for_memory()
    freeze = OUTPUT / "frozen.json"
    assert not git("status","--porcelain"), "Commit the frozen configuration and code before running test"
    # Verify the configuration is actually included in HEAD.
    git("ls-files","--error-unmatch",str(freeze.relative_to(ROOT)))
    config = json.loads(freeze.read_text())
    stamp = dict(commit=git("rev-parse","HEAD"), config_sha256=digest(freeze), git_dirty=False)
    base = CACHE / "locked_test"
    base.mkdir(parents=True, exist_ok=True)
    stamp_path = base / "provenance.json"
    if stamp_path.exists():
        assert json.loads(stamp_path.read_text()) == stamp, "Cannot resume a different configuration"
    else:
        dump(stamp_path, stamp)
    assert len(config["models"]) == 6
    for m in config["models"]:
        assert digest(weights(m["model"], m["seed"])) == m["weights_sha256"]
    assert digest(MANIFEST / "test.csv") == config["manifests"]["test"]
    french = [r["image_id"] for r in manifest("test")]
    polygons = match("test")
    levir_list = ROOT / "data/manifests/test_filenames.txt"
    assert digest(levir_list) == config["levir_test_manifest"]
    levir = levir_list.read_text().split()
    assert set(polygons) == set(french) and len(french) == len(set(french)) == 979
    assert len(set(levir)) == len(levir) == 128
    for dataset, names in (("fotbcd", french), ("levir", levir)):
        rows = []
        for family in ("A","S"):
            for seed in SEEDS:
                rows.extend(evaluate_rows(dataset, family, seed, config, stamp, names, polygons))
        write(base / f"{dataset}_counts.csv", rows)
    print("Test complete", stamp, flush=True)


if __name__ == "__main__":
    main()
