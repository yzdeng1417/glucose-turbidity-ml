# Data and methods

## Synthetic inputs

The default `calibrated` example is a fixed synthetic sample packaged under [`src/cmp/data/calibrated-v1`](../src/cmp/data/calibrated-v1), with separate development and held-out groups. `cmp.example_data.load_demo_data` checks the file hashes and group split before use. Changing the CLI seed changes training and grouped cross-validation, not the packaged rows, labels or held-out membership.

The sample is synthetic and calibrated to aggregate properties of a pure-culture development dataset. Its distribution reflects pooled marginal quantiles, contextual label probabilities, group/series dependence and a common four-factor model for paired signals. Identities, labels and trajectories are newly sampled rather than copied experimental curves. Experimental measurements, fitted calibration parameters and real-versus-synthetic comparison figures are not included. Running the demo reproduces analysis of this fixed sample; it does not regenerate the calibrated data.

Each group has one species category and observations across drug contexts and concentration levels. Combination contexts such as `amp_sul` and `amk_lev` retain both drug indicators. Additional schema columns can be inactive in this sample. Signal times are 30, 45, 60, 75, 90, 105, 120 and 135 minutes. Values retain input-table numerical units; the names do not establish physical glucose or NTU calibration. Concentration levels are context-specific ordinal codes, not a common physical concentration scale across drugs.

All labels have provenance `simulated_endpoint_label`; none are measured outcomes. Level 0 is fixed to label 1 as a simulation design choice. Group IDs and row IDs are newly generated, while species and drug names describe categories, not patient or isolate identities. Marginal tails and sparse strata are deliberately simplified. Within-group dose-response detail and time-point variation are imperfectly preserved. The synthetic held-out set demonstrates software separation under this generator, not clinical generalization or reproduction of a study. No formal privacy guarantee is claimed.

`--dataset toy` provides a smaller equation-based example through `cmp.synthetic.make_synthetic`. It reads no experimental files and estimates no experimental parameters. Artificial species, drugs and concentration levels drive a toy latent response; Bernoulli labels and noisy paired signals depend on it. Its time indices are arbitrary, with derivative step 1.0. The seed controls both toy generation and training. These equations are simulation assumptions, not conclusions about bacterial physiology.

## Input schema

The model reads a feature TSV and keeps metadata separately.

| Columns | Meaning |
|---|---|
| `species_*` | Exactly one active category per row |
| `antibiotic_*` | One or more active context indicators; excludes the concentration prefix |
| `antibiotic_concentration_*` | Exactly one active concentration-level indicator |
| `glu_time1` … `glu_timeN` | Glucose-channel trajectory |
| `ntu_time1` … `ntu_timeN` | Paired turbidity-channel trajectory |
| `growth` | Binary label, exactly 0 or 1 |

Time columns must be contiguous and are ordered numerically; paired channels must have equal length, with at least three points. Values must be finite. Unknown columns, duplicate headers, invalid labels and malformed indicators are rejected. Combination-drug indicators are supported and exercised by the default sample.

The command-line entry point uses synthetic inputs only. The packaged metadata use `synthetic_split`; the adapter renames that field to `split` in run outputs without changing any membership. Row/group/series IDs, decoded categories and label provenance are kept out of model inputs. The data loader can be imported for other properly authorized datasets, but that requires a separately specified protocol, group identities, label provenance and validation plan. This demo does not define those choices for real experiments.

## Representations

For one row and one channel with values `x[t]` and uniform interval `dt`:

- `raw`: `x[t]`.
- `centered`: `x[t] - x[0]`.
- `first_derivative`: `(x[t+1] - x[t]) / dt`.
- `second_derivative`: `(x[t+2] - 2*x[t+1] + x[t]) / dt**2`.

Transforms are computed in float64 and converted to float32 model inputs. The CLI uses `dt=15` minutes for the calibrated example and `dt=1` arbitrary step for the optional toy. Centering removes a constant additive offset; it does not correct every source of interference. Short finite differences can amplify noise. No representation is declared superior here.

## Model and training

The baseline is a scikit-learn pipeline with `StandardScaler` and logistic regression (`C=1`, maximum 1,000 iterations). Both are fitted only on the training portion of each grouped fold. The final baseline uses all development rows.

The neural model has tabular towers for the three categorical blocks and convolutional towers for the two time-series blocks. Mean/max pooling feeds block embeddings into attention fusion and a binary-logit head. Neural scaling applies to time-series blocks only. Optimization uses class-weighted binary cross-entropy and AdamW; the stopping criterion is ROC-AUC on a separate group partition. The final neural model also retains a development-only stopping partition rather than fitting on every development row. The demo uses one model per fold and one final model, not an ensemble.

All neural configuration values are recorded in `run_manifest.json`. CPU execution uses one PyTorch thread and deterministic algorithms. Seeds and dependency versions are recorded, but bitwise identity across operating systems, library versions or hardware is not promised. Checkpoints are generated locally from synthetic inputs; no checkpoint is downloaded or loaded by the example.

## Threshold and reporting

Candidate thresholds cover all attainable binary decisions from the development OOF scores, including all-negative predictions. Balanced accuracy is maximized; ties favor proximity to 0.5, then the lower threshold. A threshold can exceed 1 by one floating-point step to encode an all-negative decision. The operating threshold is selected from the synthetic development scores for each run.

The saved split audit allows reviewers to check group separation, OOF coverage and neural early-stopping roles. Its row identifiers refer only to generated data. The manifest hashes code, configuration-bearing outputs, predictions and inputs; it does not contain local usernames or absolute workspace paths. It excludes its own hash to avoid self-reference.

The report contains illustrative trajectories, a held-out ROC and a confusion matrix. No clinical confidence intervals, MIC estimates, resistance categories or comparative superiority claims are produced. Observations within a synthetic group are correlated, just as repeated rows can be correlated in a real study; row count must not be confused with the number of independent groups.

## Biological and clinical interpretation

The model predicts a binary growth endpoint for an individual observation. In experimental applications, that endpoint must be defined by the reference assay and observation time. A growth prediction alone does not determine bacterial survival, killing or treatment success. Glucose-related activity and turbidity can provide different descriptions of a response; any claim that combining them improves prediction requires matched comparisons with each channel alone.

An MIC estimate requires a valid concentration series, growth controls and an explicit endpoint rule. Clinical susceptibility interpretation then requires the applicable organism–drug breakpoints and exposure assumptions. The demo produces neither MIC estimates nor clinical susceptibility categories.

The future purpose is to provide clinicians with useful evidence for patient-specific antibiotic selection. Establishing that value requires independent validation of measurements and decision errors, evaluation alongside reference testing and clinical information, and evidence of benefit to patient care. The signal-sampling window alone does not establish a shorter clinical turnaround time, and synthetic classification metrics cannot demonstrate patient benefit.
