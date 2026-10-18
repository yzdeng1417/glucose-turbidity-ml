# Data and methods

## Synthetic inputs

`cmp.synthetic.make_synthetic` uses a seeded NumPy generator and equations written for this demonstration. It reads no local study files, fetches no remote datasets and estimates no parameters from experiments. The data are simulated from scratch, rather than anonymized experimental records.

Each group has one artificial species category and shared random offsets. Its eight observations cover two artificial drug categories and four concentration levels. A toy latent response depends on concentration, drug category and a group effect. A Bernoulli draw supplies the binary label; noisy trajectories depend on the same latent response. Glucose-like signals decrease and turbidity-like signals increase with that response by construction. Those choices are simulation assumptions, not conclusions about bacterial physiology.

There are eight equally spaced toy time indices in arbitrary units. The default derivative step is 1.0. Category names A/B/C, ordinal concentration levels and all group IDs are fictitious. Neither signal units nor concentration levels can be interpreted as physical measurements. The toy generator does not model species diversity, sample matrices, realistic instruments, assay calibration or clinical reference methods.

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

Time columns must be contiguous and are ordered numerically; paired channels must have equal length, with at least three points. Values must be finite. Unknown columns, duplicate headers, invalid labels and malformed indicators are rejected. Combination-drug indicators are allowed by the interface, although this toy dataset uses single categories.

The command-line entry point always generates synthetic inputs. The data loader can be imported for other properly authorized datasets, but that requires a separately specified protocol, group identities, label provenance and validation plan. This demo does not define those choices for real experiments.

## Representations

For one row and one channel with values `x[t]` and uniform interval `dt`:

- `raw`: `x[t]`.
- `centered`: `x[t] - x[0]`.
- `first_derivative`: `(x[t+1] - x[t]) / dt`.
- `second_derivative`: `(x[t+2] - 2*x[t+1] + x[t]) / dt**2`.

Transforms are computed in float64 and converted to float32 model inputs. Centering removes a constant additive offset; it does not correct every source of interference. Short finite differences can amplify noise. No representation is declared superior here.

## Model and training

The baseline is a scikit-learn pipeline with `StandardScaler` and logistic regression (`C=1`, maximum 1,000 iterations). Both are fitted only on the training portion of each grouped fold. The final baseline uses all development rows.

The neural model has tabular towers for the three categorical blocks and convolutional towers for the two time-series blocks. Mean/max pooling feeds block embeddings into attention fusion and a binary-logit head. Neural scaling applies to time-series blocks only. Optimization uses class-weighted binary cross-entropy and AdamW; the stopping criterion is ROC-AUC on a separate group partition. The final neural model also retains a development-only stopping partition rather than fitting on every development row. The demo uses one model per fold and one final model, not an ensemble.

All neural configuration values are recorded in `run_manifest.json`. CPU execution uses one PyTorch thread and deterministic algorithms. Seeds and dependency versions are recorded, but bitwise identity across operating systems, library versions or hardware is not promised. Checkpoints are generated locally from synthetic inputs; no checkpoint is downloaded or loaded by the example.

## Threshold and reporting

Candidate thresholds cover all attainable binary decisions from the development OOF scores, including all-negative predictions. Balanced accuracy is maximized; ties favor proximity to 0.5, then the lower threshold. A threshold can exceed 1 by one floating-point step to encode an all-negative decision. No historical study threshold is bundled.

The saved split audit allows reviewers to check group separation, OOF coverage and neural early-stopping roles. Its row identifiers refer only to generated data. The manifest hashes code, configuration-bearing outputs, predictions and inputs; it does not contain local usernames or absolute workspace paths. It excludes its own hash to avoid self-reference.

The report contains illustrative trajectories, a held-out ROC and a confusion matrix. No clinical confidence intervals, MIC estimates, resistance categories or comparative superiority claims are produced. Observations within a synthetic group are correlated, just as repeated rows can be correlated in a real study; row count must not be confused with the number of independent groups.

## Relation to the research code

The exported model architecture and selected scaling, training and evaluation functions preserve the corresponding research implementation's algorithms. Public adapters remove private protocol paths and dataset-specific assumptions. Input validation is strengthened, time spacing is configurable in the transformation API, and the demonstration uses newly specified toy settings. Documentation comments describing historical study performance have been removed. This release does not claim that the demo's configuration or metrics reproduce a research result.

The original dataset, experimental labels, study protocols, fitted weights and manuscript assets are outside this release. Reproducing a future publication would require its separately approved data and protocol package.
