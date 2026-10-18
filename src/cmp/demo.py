"""Reproducible CLI using only independently generated synthetic inputs."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np

from cmp.evaluation import metrics
from cmp.plotting import plot_report
from cmp.representations import REPRESENTATIONS, load_representation
from cmp.synthetic import make_synthetic
from cmp.workflow import develop, score_model, subset, neural_protocol


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(output, *, seed=2026, model="baseline", representation="centered", folds=3, epochs=15):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new output directory")
    if model not in ("baseline", "neural") or representation not in REPRESENTATIONS:
        raise ValueError("Unknown model or representation")
    if folds < 2 or folds > 36 or epochs < 1 or seed < 0 or seed > 2**32 - 100:
        raise ValueError("Invalid folds, epochs or seed")
    if model == "neural":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError('Install the neural extra first: pip install -e ".[neural]"') from exc
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True)
    raw, metadata = make_synthetic(seed=seed)
    raw.to_csv(output / "synthetic_features.tsv", sep="\t", index=False, float_format="%.12g")
    metadata.to_csv(output / "synthetic_metadata.tsv", sep="\t", index=False)
    arrays = load_representation(output / "synthetic_features.tsv", representation)
    development_indices = np.flatnonzero(metadata.split.eq("development"))
    heldout_indices = np.flatnonzero(metadata.split.eq("heldout"))
    groups = metadata.group_id.to_numpy()
    if set(groups[development_indices]) & set(groups[heldout_indices]):
        raise ValueError("Development / holdout group overlap")
    development = subset(arrays, development_indices)
    scores, fold_id, threshold, _, fitted, stats, audit = develop(
        development, groups[development_indices], model, seed, folds, epochs, output,
    )
    audit["row_id"] = metadata.iloc[development_indices].row_id.to_numpy()[audit.development_row_index]
    audit.to_csv(output / "split_audit.tsv", sep="\t", index=False)
    threshold_record = {"value": threshold, "source": "pooled_grouped_development_oof",
                        "objective": "balanced_accuracy", "heldout_used": False,
                        "development_operating_metrics": "threshold-selection diagnostics"}
    # Persist threshold before any held-out prediction or metric computation.
    write_json(output / "threshold.json", threshold_record)
    heldout = subset(arrays, heldout_indices)
    heldout_scores = score_model(fitted, stats, heldout, model)
    frames = {}
    for label, indices, prediction in (
        ("development_oof", development_indices, scores),
        ("synthetic_holdout", heldout_indices, heldout_scores),
    ):
        frame = metadata.iloc[indices].reset_index(drop=True).copy()
        frame["growth"] = arrays.y[indices]
        frame["score"] = prediction
        frame["prediction"] = (prediction >= threshold).astype(int)
        if label == "development_oof":
            frame["outer_fold"] = fold_id
        frame.to_csv(output / f"{label}.tsv", sep="\t", index=False, float_format="%.12g")
        frames[label] = frame
    result = {"data_kind": "synthetic", "threshold": threshold_record,
              "development_oof_selection_diagnostics": metrics(frames["development_oof"], threshold),
              "synthetic_holdout": metrics(frames["synthetic_holdout"], threshold),
              "interpretation": "Software demonstration only; no experimental or clinical performance claim."}
    write_json(output / "metrics.json", result)
    plot_report(raw.iloc[development_indices], frames["synthetic_holdout"], threshold, output / "report.png")
    package_dir = Path(__file__).parent
    packages = ["glucose-turbidity-ml", "numpy", "pandas", "scikit-learn", "matplotlib", "PyYAML"]
    if model == "neural":
        packages.append("torch")
    manifest = {
        "data_kind": "synthetic", "generator": "cmp.synthetic.make_synthetic",
        "configuration": {"seed": seed, "model": model, "representation": representation,
                          "folds": folds, "epochs": epochs if model == "neural" else None,
                          "time_step": 1.0, "time_unit": "arbitrary toy step",
                          "neural_protocol": neural_protocol(epochs) if model == "neural" else None},
        "counts": {"development_rows": len(development_indices), "heldout_rows": len(heldout_indices),
                   "development_groups": len(set(groups[development_indices])), "heldout_groups": len(set(groups[heldout_indices]))},
        "environment": {"python": platform.python_version(), "system": platform.system(),
                        "machine": platform.machine(), "packages": {p: version(p) for p in packages}},
        "source_sha256": {str(p.relative_to(package_dir)): digest(p) for p in sorted(package_dir.rglob("*")) if p.suffix in (".py", ".yaml")},
        "artifact_sha256": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    write_json(output / "run_manifest.json", manifest)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model", choices=("baseline", "neural"), default="baseline")
    parser.add_argument("--representation", choices=REPRESENTATIONS, default="centered")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=15, help="Neural training only")
    args = parser.parse_args()
    try:
        result = run(args.output, seed=args.seed, model=args.model, representation=args.representation,
                     folds=args.folds, epochs=args.epochs)
    except (ValueError, FileExistsError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps({"data_kind": "synthetic", "output": str(args.output),
                      "threshold": result["threshold"]["value"], "status": "complete"}, indent=2))


if __name__ == "__main__":
    main()
