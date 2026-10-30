"""Recompute or retrain the frozen real-data, centered paper model."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from cmp.evaluation import metrics
from cmp.paper_data import DATA_DIR, SCORE_COLUMNS, SEEDS, digest, load_data, read_table
from cmp.paper_mic import mic_tables


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def save_table(frame, path):
    frame.to_csv(path, sep="\t", index=False, float_format="%.17g", lineterminator="\n")


def environment():
    versions = {name: importlib.metadata.version(package) for name, package in
                (("numpy", "numpy"), ("pandas", "pandas"), ("sklearn", "scikit-learn"))}
    versions["python"] = platform.python_version()
    try:
        versions["torch"] = importlib.metadata.version("torch").split("+")[0]
    except importlib.metadata.PackageNotFoundError:
        versions["torch"] = None
    return versions


def train(bundle, output):
    import torch
    from cmp.paper_training import fit_member, predict, subset

    actual = environment()
    expected = bundle["protocol"]["environment"]
    if actual != expected:
        raise ValueError(f"Strict training requires the reference environment: {expected}; found {actual}. See requirements-paper.txt")
    torch.set_num_threads(bundle["protocol"]["training"]["torch_threads"])
    torch.use_deterministic_algorithms(True)
    development = bundle["cohorts"]["development"]["arrays"]
    balf = bundle["cohorts"]["balf"]["arrays"]
    (output / "models").mkdir()
    scores = {"development": np.full((1605, 5), np.nan), "balf": np.full((168, 5), np.nan)}
    folds = np.zeros(1605, dtype=int)
    records = []
    for fold, seed, fit, stop, evaluation in bundle["plan"]:
        started = time.monotonic()
        name = f"outer_{fold}_seed_{seed}" if fold else f"final_seed_{seed}"
        model, stats, record = fit_member(development, fit, stop, seed, bundle["protocol"],
                                          output / "models" / f"{name}.pt")
        cohort = "development" if fold else "balf"
        target = subset(development, evaluation) if fold else balf
        score = predict(model, stats, target, bundle["protocol"]["training"]["batch_size"])
        row_indices = evaluation if fold else np.arange(168)
        scores[cohort][row_indices, SEEDS.index(seed)] = score
        if fold:
            folds[evaluation] = fold
        record.update(stage="outer" if fold else "final", fold=fold, seconds=time.monotonic() - started)
        records.append(record)
        save_table(pd.DataFrame(records), output / "fit_summary.tsv")
        print(f"{name}: best epoch {record['best_epoch']}, {record['epochs_run']} epochs ({record['seconds']:.1f}s)", flush=True)
    predictions = {}
    for cohort, matrix in scores.items():
        if not np.isfinite(matrix).all():
            raise ValueError("Incomplete predictions")
        c = bundle["cohorts"][cohort]
        frame = c["metadata"][["row_index0", "row_id", "group_id"]].copy()
        frame["growth"] = c["arrays"].y
        # Each predict() casts the sigmoid float32 scores to float64 before averaging.
        frame["score"] = matrix.mean(axis=1)
        for i, seed in enumerate(SEEDS):
            frame[f"member_score_{seed}"] = matrix[:, i]
        frame["threshold"], frame["prediction"] = .5, (frame.score >= .5).astype(int)
        if cohort == "development":
            frame["outer_fold"] = folds
        predictions[cohort] = frame
    return predictions, pd.DataFrame(records)


def classification_metrics(frames):
    records = []
    for cohort, frame in frames.items():
        populations = {"all": frame, "drug_exposed": frame.loc[frame.concentration_level.ne(0)],
                       "no_drug_control": frame.loc[frame.concentration_level.eq(0)]}
        for population, rows in populations.items():
            result = metrics(rows, .5)
            result["accuracy"] = (result["tn"] + result["tp"]) / result["n"]
            result["false_positive_rate"] = result["fp"] / result["no_growth"] if result["no_growth"] else None
            records.append({"cohort": cohort, "population": population, "groups": rows.group_id.nunique(), **result})
    return pd.DataFrame(records)


def compare_records(actual, expected, keys):
    actual = actual.set_index(keys).sort_index()
    expected = expected.set_index(keys).sort_index()
    if not actual.index.equals(expected.index):
        return {"matched": False, "changed_records": max(len(actual), len(expected)), "reason": "series coverage differs"}
    changed = set()
    columns = []
    for column in expected.columns:
        if column not in actual:
            return {"matched": False, "changed_records": len(expected), "reason": f"missing column {column}"}
        for key, a, b in zip(expected.index, actual[column], expected[column]):
            if pd.isna(a) and b == "":
                continue
            try:
                equal = abs(float(a) - float(b)) <= 1e-12
            except (ValueError, TypeError):
                equal = str(a) == str(b)
            if not equal:
                changed.add(str(key))
                if column not in columns:
                    columns.append(column)
    return {"matched": not changed, "changed_records": len(changed), "changed_keys": sorted(changed), "changed_columns": columns}


def reconcile(bundle, predictions, fits, mic, metric_table):
    checks = {}
    for cohort, frame in predictions.items():
        expected = bundle["cohorts"][cohort]["saved"]
        delta = float(np.abs(frame[SCORE_COLUMNS].to_numpy() - expected[SCORE_COLUMNS].to_numpy()).max())
        changed = int((frame.prediction.to_numpy() != expected.prediction.to_numpy()).sum())
        checks[cohort + "_scores"] = {"max_absolute_delta": delta, "changed_calls": changed,
            "matched": delta <= bundle["policy"]["score_max_absolute_error"] and changed == 0,
            "per_column_max_absolute_delta": {column: float(np.abs(frame[column] - expected[column]).max()) for column in SCORE_COLUMNS}}
    for cohort, table in zip(("development", "balf"), mic):
        reference = read_table(bundle["directory"] / f"{cohort}_expected_mic.tsv")
        checks[cohort + "_mic"] = compare_records(table, reference, ["group_id", "drug_context"])
    expected_frames = {name: c["metadata"].merge(c["saved"].drop(columns=["group_id", "row_index0"]),
                       on="row_id", validate="one_to_one") for name, c in bundle["cohorts"].items()}
    expected_metrics = classification_metrics(expected_frames)
    metrics_delta = 0.0
    for column in expected_metrics.select_dtypes(include="number"):
        a, b = metric_table[column].to_numpy(dtype=float), expected_metrics[column].to_numpy(dtype=float)
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            metrics_delta = float("inf")
            break
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.any():
            metrics_delta = max(metrics_delta, float(np.abs(a[finite] - b[finite]).max()))
    checks["classification_metrics"] = {"matched": metrics_delta <= bundle["policy"]["classification_metric_absolute_error"],
                                          "max_absolute_delta": metrics_delta if np.isfinite(metrics_delta) else None}
    if fits is not None:
        expected = read_table(bundle["directory"] / "expected_fit_summary.tsv")
        checks["fit_epochs_and_stopping_auc"] = compare_records(fits, expected, ["stage", "fold", "seed"])
    return {"passed": all(check["matched"] for check in checks.values()), "checks": checks}


def write_report(path, mode, metric_table, mic_summary, reconciliation):
    lines = ["# Real-data paper reproduction", "", f"Mode: `{mode}`. Reconciliation: **{'PASS' if reconciliation['passed'] else 'FAIL'}**.", "",
             "`saved` recomputes from frozen predictions; `train` fits all 30 paper models and compares their outputs.", "",
             "The BALF cohort is a previously examined retrospective feasibility cohort (8 specimens, 6 participant groups).",
             "The fixed 0.5 threshold was adopted after reviewing development/BALF results. Scores are uncalibrated.", "",
             "| Cohort | Subset | n | TN | FP | FN | TP | ROC AUC |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in metric_table.to_dict("records"):
        auc = f"{row['roc_auc']:.6f}" if pd.notna(row["roc_auc"]) else "NA"
        lines.append(f"| {row['cohort']} | {row['population']} | {row['n']} | {row['tn']} | {row['fp']} | {row['fn']} | {row['tp']} | {auc} |")
    pure, balf = mic_summary["development"], mic_summary["balf"]
    lines += ["", f"Pure cultures: {pure['reportable']}/{pure['series']} reportable, {pure['numeric_pairs']} numeric pairs; "
              f"EA {pure['ea_n']}/{pure['numeric_pairs']}, exact {pure['exact_n']}/{pure['numeric_pairs']}.",
              f"BALF: {balf['reportable']}/{balf['samples']} reportable; EA {balf['ea_n']}/{balf['comparable']}, "
              f"range-adjusted exact {balf['exact_after_offscale_n']}/{balf['comparable']}.", "",
              "All series, range qualifiers, NR reasons and exclusion denominators are retained in the MIC tables.",
              "Range-adjusted equality describes compatibility of reported bounds, not known identical exact MICs.", ""]
    path.write_text("\n".join(lines))


def run(mode, output, data_dir=DATA_DIR):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new output directory; existing evidence is never overwritten")
    bundle = load_data(data_dir)
    output.mkdir(parents=True)
    save_table(bundle["splits"], output / "split_manifest.tsv")
    write_json(output / "protocol.json", bundle["protocol"])
    write_json(output / "reconciliation_policy.json", bundle["policy"])
    manifest = {"mode": mode, "status": "running", "dataset_id": bundle["manifest"]["dataset_id"],
                "synthetic": False, "threshold": .5, "environment": environment(), "platform": platform.platform(),
                "dataset_manifest_sha256": digest(bundle["directory"] / "manifest.json"),
                "input_sha256": bundle["manifest"]["files"], "started_at_unix": time.time(),
                "package_version": importlib.metadata.version("glucose-turbidity-ml"),
                "source_sha256": {str(p.relative_to(Path(__file__).parent)): digest(p)
                                  for p in sorted(Path(__file__).parent.rglob("*.py"))}}
    write_json(output / "run_manifest.json", manifest)
    try:
        if mode == "train":
            predictions, fits = train(bundle, output)
        else:
            predictions = {name: c["saved"].copy() for name, c in bundle["cohorts"].items()}
            fits = None
        frames = {}
        for name, prediction in predictions.items():
            frames[name] = bundle["cohorts"][name]["metadata"].merge(
                prediction.drop(columns=["group_id", "row_index0"]), on="row_id", validate="one_to_one")
            save_table(frames[name], output / f"{name}_predictions.tsv")
        metric_table = classification_metrics(frames)
        save_table(metric_table, output / "classification_metrics.tsv")
        fold_metrics = []
        for fold, frame in frames["development"].groupby("outer_fold"):
            for record in classification_metrics({"development": frame}).to_dict("records"):
                fold_metrics.append({"outer_fold": fold, **record})
        save_table(pd.DataFrame(fold_metrics), output / "development_fold_metrics.tsv")
        pure, balf, votes, summary = mic_tables(frames["development"], frames["balf"], bundle["comparators"])
        save_table(pure, output / "development_mic.tsv")
        save_table(balf, output / "balf_mic.tsv")
        save_table(votes, output / "balf_concentration_votes.tsv")
        write_json(output / "mic_summary.json", summary)
        reconciliation = reconcile(bundle, predictions, fits, (pure, balf), metric_table)
        write_json(output / "reconciliation.json", reconciliation)
        write_report(output / "report.md", mode, metric_table, summary, reconciliation)
        manifest.update(status="passed" if reconciliation["passed"] else "mismatch", models_trained=30 if mode == "train" else 0)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at_unix"] = time.time()
        manifest["output_sha256"] = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*"))
                                     if p.is_file() and p.name != "run_manifest.json"}
        write_json(output / "run_manifest.json", manifest)
    if not reconciliation["passed"]:
        raise RuntimeError(f"Reconciliation failed; retained outputs in {output}")
    return {"mode": mode, "status": "passed", "models_trained": manifest["models_trained"], "mic": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("saved", "train"), default="saved")
    parser.add_argument("--output", type=Path, default=Path("outputs/paper-saved"))
    args = parser.parse_args()
    try:
        result = run(args.mode, args.output)
    except (ValueError, FileExistsError, RuntimeError, ImportError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
