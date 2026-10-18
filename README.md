# Glucose–turbidity analysis of bacterial drug responses

[![Checks](https://github.com/yzdeng1417/glucose-turbidity-ml/actions/workflows/checks.yml/badge.svg)](https://github.com/yzdeng1417/glucose-turbidity-ml/actions/workflows/checks.yml)

An open-source Python toolkit for studying bacterial responses to antibiotics through paired glucose (GLU) and turbidity (NTU) time series. Its long-term goal is to support better-informed antibiotic choices for individual patients, helping them receive effective treatment while avoiding unnecessary drug exposure.

The toolkit connects paired response trajectories with growth-endpoint classification, using species, drug and concentration context. A runnable synthetic example demonstrates the analysis from input checks to grouped evaluation.

## Biological rationale

Glucose utilization and changes in culture turbidity describe different aspects of bacterial physiology:

- **Glucose-related signals** provide a view of substrate utilization when interpreted with the assay's calibration and controls. Glucose use and increases in culture turbidity need not change together, which motivates studying both during antibiotic exposure.
- **Turbidity signals** describe the optical cloudiness of a suspension and are commonly used to follow culture growth. Their relationship to cell abundance depends on measurement conditions and cell properties, including size; they are not direct counts of viable bacteria.

Paired measurements let researchers ask whether a drug affects glucose-related activity and bulk growth in different ways, and whether those differences help interpret a later growth endpoint. A difference between the signals is a candidate phenotype to investigate; establishing survival, killing or a specific resistance mechanism requires additional evidence.

## Research applications and future patient care

Researchers can use the analysis components to examine paired trajectories across drug contexts, compare signal representations and evaluate growth-endpoint predictions while keeping repeated observations from the same group together. Applying them to experimental data requires a defined assay, reference labels and validation protocol.

The intended clinical direction is phenotypic antimicrobial susceptibility testing: learning how a patient's infecting organism responds to candidate antibiotics, then evaluating whether that information can help clinicians choose appropriate treatment. For patients, the hoped-for value is less uncertainty about treatment, less exposure to ineffective or unnecessary antibiotics, and fewer avoidable medication-related harms.

Reaching that goal requires independent experimental and clinical validation, comparison with reference susceptibility methods, and assessment of treatment decisions, patient outcomes and the full time from sample receipt to a usable report. The current software does not establish faster testing or improved patient outcomes.

**Current example:** all included observations, labels and plotted results are synthetic. The prediction target is a **binary growth label per observation**, separate from a minimum inhibitory concentration (MIC) estimate or a clinical susceptibility category. The example demonstrates research software and is not validated for treatment decisions.

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

The default uses a [calibrated synthetic example](src/cmp/data/calibrated-v1) with separate development and held-out groups. Its rows, labels and split stay the same across runs; `--seed` controls cross-validation and model training. File checksums are verified before use. No dataset download or local calibration is required.

For a minimal installation without development tools, use `python -m pip install -e .`. The requirements file pins the tested environment, including test tools; `pyproject.toml` declares compatible dependency ranges.

## Software components

| Component | What it demonstrates | Code |
|---|---|---|
| Input contract | Validate labels, finite values, categorical encoding and ordered time points | [dataset.py](src/cmp/dataset.py), [feature_schema.yaml](src/cmp/feature_schema.yaml) |
| Signal representations | Raw trajectories, within-row baseline subtraction and finite differences | [representations.py](src/cmp/representations.py) |
| Group-aware evaluation | Keep repeated observations from each group together; fit preprocessing inside training folds | [workflow.py](src/cmp/workflow.py) |
| Neural model | Separate tabular/temporal encoders and block-attention fusion | [multitower.py](src/cmp/models/multitower.py), [neural.py](src/cmp/neural.py) |
| Reproducibility | Save predictions, split membership, configuration, dependency versions and SHA-256 hashes | [demo.py](src/cmp/demo.py) |
| Fixed synthetic example | Verify packaged synthetic observations and preserve their group split | [example_data.py](src/cmp/example_data.py), [dataset manifest](src/cmp/data/calibrated-v1/manifest.json) |
| Verification | Check leakage boundaries, input failures, transforms and complete runs | [tests/](tests/) |

The lightweight default is standardized logistic regression. It provides a quick way to inspect the evaluation workflow. The optional multi-tower neural model combines categorical context with both signal trajectories, with separate groups for optimization, early stopping and outer evaluation.

```bash
# Optional CPU neural example
python -m pip install -r requirements-neural.txt
glu-ntu-demo --model neural --epochs 15 --output outputs/neural

# Explore a different signal representation
glu-ntu-demo --representation first_derivative --output outputs/first-derivative

# Optional smaller, equation-based toy example
glu-ntu-demo --dataset toy --seed 2026 --output outputs/toy

# Run checks (neural tests are skipped when PyTorch is absent)
python -m pytest -q
python -m ruff check .
```

## Evaluation design

```text
Synthetic example
  ├── Development groups
  │     ├── Grouped out-of-fold predictions
  │     │     └── select threshold by balanced accuracy
  │     └── fit final model using development data
  └── Held-out synthetic groups
        └── score after development and threshold selection
```

For the neural model, each training portion is split again by group for early stopping. Time-series scaling is fitted only on optimizer rows. For the baseline, the scaler and classifier are fitted together on each training fold. Within-row centering is a separate operation that subtracts each signal's first value. Derivative representations use the sampling interval recorded for the dataset; the optional toy example uses arbitrary time steps. See [data and methods](docs/data-and-methods.md) for the time axis and transformation definitions.

The operating threshold is selected from pooled development out-of-fold scores, then saved before held-out prediction. Metrics on those same development scores are **threshold-selection diagnostics**, not unbiased estimates of the selected operating point. The held-out synthetic set demonstrates separation in this example; it does not establish generalization to real samples. Comparing repeated runs or choosing representations after viewing held-out results would introduce further model-selection bias.

Scores are not presented as calibrated growth probabilities. Attention weights are internal model quantities and do not establish causal effects or the incremental value of combining channels. This repository is a research software example, not a clinical decision tool.

## Outputs

| File | Contents |
|---|---|
| `synthetic_features.tsv` | Packaged synthetic categorical indicators, paired signals and simulated labels; generated on demand only with `--dataset toy` |
| `synthetic_metadata.tsv` | Synthetic row/group IDs and development/holdout membership, kept out of model inputs |
| `split_audit.tsv` | Role of every development row in every fold; fold 0 is the final fit |
| `threshold.json` | Selected value, selection objective and data source |
| `development_oof.tsv`, `synthetic_holdout.tsv` | Labels, scores and predictions with synthetic IDs |
| `metrics.json`, `report.png` | Evaluation summaries, explicitly marked synthetic |
| `run_manifest.json` | Dataset provenance/checksums, time spacing, configuration, software versions, code hashes and output hashes |
| `*.pt`, `*.history.tsv` | Neural-only checkpoints and early-stopping histories, generated locally |

See [data and methods](docs/data-and-methods.md) for synthetic-data provenance, the optional toy generator, schema and interpretation limits.

## License

[MIT](LICENSE).
