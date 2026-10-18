# Data and methods

## Synthetic inputs

The default `calibrated` example is a fixed, reviewed synthetic sample packaged under [`src/cmp/data/calibrated-v1`](../src/cmp/data/calibrated-v1). It contains 2,253 observations from 60 new synthetic groups: 1,794 rows / 48 groups for development and 459 rows / 12 groups held out. `cmp.example_data.load_demo_data` checks the file hashes and group split before use. Changing the CLI seed changes training and grouped cross-validation, not the packaged rows, labels or held-out membership.

The sample was synthesized locally using aggregate properties fitted on a pure-culture development dataset. Generation used pooled marginal quantiles, contextual label probabilities, group/series dependence and a common four-factor procedure for paired signals. New identities, labels and trajectories were sampled; experimental observations were not used as curve donors. This is data-calibrated synthesis, not a wholly data-independent simulation. The published files contain only the approved synthetic sample and public metadata. The original measurements, fitted calibration parameters and real-versus-synthetic comparisons are not distributed. Re-running this demo reproduces downstream analysis of the frozen sample; it does not refit or regenerate the calibrated generator.

Each group has one species category and observations across drug contexts and concentration levels. The example includes five active species categories, ten complete drug contexts and ten possible level codes. Combination contexts such as `amp_sul` and `amk_lev` retain both drug indicators. Additional schema columns can be inactive in this sample. Signal times are 30, 45, 60, 75, 90, 105, 120 and 135 minutes. Values retain input-table numerical units; the names do not establish physical glucose or NTU calibration. Concentration levels are context-specific ordinal codes, not a common physical concentration scale across drugs.

All labels have provenance `simulated_endpoint_label`; none are measured outcomes. Level 0 is fixed to label 1 as a simulation design choice. Group IDs and row IDs are newly generated, while species and drug names describe categories, not patient or isolate identities. Marginal tails and sparse strata are deliberately simplified. Within-group dose-response detail and time-point variation are imperfectly preserved. The synthetic held-out set demonstrates software separation under this generator, not clinical generalization or reproduction of a study. No formal privacy guarantee is claimed.

`--dataset toy` retains the smaller equation-based example through `cmp.synthetic.make_synthetic`. It reads no experimental files, estimates no experimental parameters and generates 48 groups with eight observations per group. Artificial species A/B/C, two artificial drugs and four levels drive a toy latent response; Bernoulli labels and noisy paired signals depend on it. Its eight time indices are arbitrary, with derivative step 1.0. The seed controls both toy generation and training. These equations are simulation assumptions, not conclusions about bacterial physiology.

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

Candidate thresholds cover all attainable binary decisions from the development OOF scores, including all-negative predictions. Balanced accuracy is maximized; ties favor proximity to 0.5, then the lower threshold. A threshold can exceed 1 by one floating-point step to encode an all-negative decision. No historical study threshold is bundled.

The saved split audit allows reviewers to check group separation, OOF coverage and neural early-stopping roles. Its row identifiers refer only to generated data. The manifest hashes code, configuration-bearing outputs, predictions and inputs; it does not contain local usernames or absolute workspace paths. It excludes its own hash to avoid self-reference.

The report contains illustrative trajectories, a held-out ROC and a confusion matrix. No clinical confidence intervals, MIC estimates, resistance categories or comparative superiority claims are produced. Observations within a synthetic group are correlated, just as repeated rows can be correlated in a real study; row count must not be confused with the number of independent groups.

## Relation to the research code

The exported model architecture and selected scaling, training and evaluation functions preserve the corresponding research implementation's algorithms. Public adapters remove private protocol paths and dataset-specific assumptions. Input validation is strengthened, time spacing is configurable in the transformation API, and the demonstration uses public example settings. Documentation comments describing historical study performance have been removed. This release does not claim that the demo's configuration or metrics reproduce a research result.

The original dataset, experimental labels, study protocols, fitted weights and manuscript assets are outside this release. Reproducing a future publication would require its separately approved data and protocol package.
