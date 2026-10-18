# Glucose–turbidity research code

[![Checks](https://github.com/yzdeng1417/glucose-turbidity-ml/actions/workflows/checks.yml/badge.svg)](https://github.com/yzdeng1417/glucose-turbidity-ml/actions/workflows/checks.yml)

A reproducible Python demonstration for learning from paired glucose and turbidity time series. It combines selected research components with an independently generated synthetic dataset, so the complete example runs without access to experimental data.

The prediction target is a **binary growth label per observation**. It is separate from an MIC estimate or a susceptible/resistant clinical category. All included examples and plotted results are synthetic; they demonstrate software behavior, not experimental performance.

![Synthetic workflow preview](docs/demo-preview.png)

## Run the example

Use Python 3.12. The default example runs on a CPU and does not require PyTorch.

```bash
git clone https://github.com/yzdeng1417/glucose-turbidity-ml.git
cd glucose-turbidity-ml
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
glu-ntu-demo --output outputs/demo
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead. `python -m cmp` is equivalent to `glu-ntu-demo`.

Open `outputs/demo/report.png` for the visual report and `outputs/demo/metrics.json` for machine-readable results. Runs refuse to overwrite an existing output directory; choose a new name when rerunning.

For a minimal installation without development tools, use `python -m pip install -e .`. The requirements file pins the tested environment, including test tools; `pyproject.toml` declares compatible dependency ranges.

## What to inspect

| Component | What it demonstrates | Code |
|---|---|---|
| Input contract | Validate labels, finite values, categorical encoding and ordered time points | [dataset.py](src/cmp/dataset.py), [feature_schema.yaml](src/cmp/feature_schema.yaml) |
| Signal representations | Raw trajectories, within-row baseline subtraction and finite differences | [representations.py](src/cmp/representations.py) |
| Group-aware evaluation | Keep repeated observations from each group together; fit preprocessing inside training folds | [workflow.py](src/cmp/workflow.py) |
| Neural model | Separate tabular/temporal encoders and block-attention fusion | [multitower.py](src/cmp/models/multitower.py), [neural.py](src/cmp/neural.py) |
| Reproducibility | Save predictions, split membership, configuration, dependency versions and SHA-256 hashes | [demo.py](src/cmp/demo.py) |
| Verification | Check leakage boundaries, input failures, transforms and complete runs | [tests/](tests/) |

The lightweight default is standardized logistic regression. It provides a quick way to inspect the evaluation workflow. The optional neural path uses the extracted multi-tower research implementation, with separate groups for optimization, early stopping and outer evaluation.

```bash
# Optional CPU neural example
python -m pip install -r requirements-neural.txt
glu-ntu-demo --model neural --epochs 15 --output outputs/neural

# Explore a different signal representation
glu-ntu-demo --representation first_derivative --output outputs/first-derivative

# Run checks (neural tests are skipped when PyTorch is absent)
python -m pytest -q
python -m ruff check .
```

## Evaluation design

```text
48 synthetic groups, 8 observations per group
  ├── 36 development groups
  │     ├── 3-fold grouped out-of-fold predictions
  │     │     └── select threshold by balanced accuracy
  │     └── fit final model using development data
  └── 12 held-out synthetic groups
        └── score after development and threshold selection
```

These counts are toy settings, not study cohort sizes. For the neural model, each training portion is split again by group for early stopping. Time-series scaling is fitted only on optimizer rows. For the baseline, the scaler and classifier are fitted together on each training fold. Within-row centering is a separate operation that subtracts each signal's first value.

The operating threshold is selected from pooled development out-of-fold scores, then saved before held-out prediction. Metrics on those same development scores are **threshold-selection diagnostics**, not unbiased estimates of the selected operating point. The held-out synthetic set demonstrates separation in this example; it does not establish generalization to real samples. Comparing repeated runs or choosing representations after viewing held-out results would introduce further model-selection bias.

Scores are not presented as calibrated growth probabilities. Attention weights are internal model quantities and do not establish causal effects or the incremental value of combining channels. This repository is a research software example, not a clinical decision tool.

## Outputs

| File | Contents |
|---|---|
| `synthetic_features.tsv` | Independently generated categorical indicators, paired signals and toy labels |
| `synthetic_metadata.tsv` | Synthetic row/group IDs and development/holdout membership, kept out of model inputs |
| `split_audit.tsv` | Role of every development row in every fold; fold 0 is the final fit |
| `threshold.json` | Selected value, selection objective and data source |
| `development_oof.tsv`, `synthetic_holdout.tsv` | Labels, scores and predictions with synthetic IDs |
| `metrics.json`, `report.png` | Evaluation summaries, explicitly marked synthetic |
| `run_manifest.json` | Configuration, software versions, code hashes and output hashes |
| `*.pt`, `*.history.tsv` | Neural-only checkpoints and early-stopping histories, generated locally |

See [data and methods](docs/data-and-methods.md) for the synthetic generator, schema, limitations and relationship to the research implementation.

## Scope and development

This is a selected code release, not a reproduction package for a manuscript. The feature-block interface, multi-tower model, neural training components and threshold enumeration originate from the research implementation. The portable demo, synthetic generator, stricter input checks and packaging adapt those components for public inspection. No experimental measurements, trained study weights, research results, manuscript text or original repository history are included.

This project was developed with AI assistance.

## License

[MIT](LICENSE).
