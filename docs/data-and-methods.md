# Paper data and reproduction protocol

The default `glu-ntu-reproduce` workflow uses the de-identified real paper dataset
`paper-centered-real-v1`. It evaluates manual endpoint growth labels, with a
separate series-level MIC calculation. It does not infer viability, killing,
tolerance, persistence or clinical susceptibility categories from the scores.

## Data and time support

The development cohort contains 1,605 observations from 43 pure-culture strains
(961 growth labels), including two reference strains. The BALF cohort contains
168 observations from eight fluid specimens (21 growth labels) from six
participant groups. These are technical observations, not 168 independent
patients. Both cohorts retain all no-drug controls, including real BALF controls
with a negative manual growth label. Labels were assigned by manual endpoint
growth reading; no single uniform endpoint incubation time is inferred here.

Each channel has eight early measurements at 30, 45, 60, 75, 90, 105, 120 and
135 minutes after addition, with 15-minute spacing. GLU denotes the recorded
glucose-assay optical signal, not a direct viable-cell count. NTU denotes the
recorded turbidity channel. Native channel units and measurement scales are
retained; no new assay calibration or concentration conversion is applied.

The feature matrices contain species indicators, drug component indicators,
concentration-level indicators, both eight-value channels and `growth` (0/1).
Metadata supplies full drug context, measured concentration (µg/mL), anonymous
row/group relations and pure-culture `reference_mic` tokens. Combination
contexts `AMP_SUL` and `AMK_LEV` remain distinct from their component drugs.
The recorded concentration describes that original context; component ratios
are not supplied and must not be inferred. In particular 6 µg/mL is retained.

`PC...` and `BF...` group IDs are opaque strain/specimen identifiers. `C...`
participant groups preserve necessary repeated-specimen relations. Calendar
dates, clinical reports, original identity keys, source paths and the internal
identity map are excluded. These IDs and all MIC reference fields are metadata,
never model inputs. The feature TSV files preserve original bytes, precision,
column order and row order. The dataset manifest contains SHA-256 hashes for
every runtime data/protocol/reference file; the loader verifies them.

## Main model

For each observation and channel, subtract its own T1 in float64, then cast to
float32, retaining the constant zero T1 column. To preserve the paper's numeric
operation order, the feature TSV is parsed using the original pandas default
parser. Standardization uses only optimizer-training rows. Per-column means
and standard deviations are calculated in float64 and stored as float32;
standard deviations below 1e-8 become 1.0. Categorical context is unchanged.

The five input blocks have separate encoders. The two temporal encoders have
two 1D convolution layers, 16 hidden channels, kernel size 3, no batch
normalization and no added derivative channel. Encoder embeddings have width
16 and dropout 0.1; fusion uses two attention heads, a 64-wide head and head
dropout 0.2. Attention weights are internal quantities, not causal effects.

The released frozen split manifest contains 48,150 row-role records: five outer
folds times five seeds, plus five final members. Seeds are 42–46. Each outer
fold holds out complete strains. Within each remaining training portion,
optimizer and early-stopping strain groups are disjoint. The historical
early-stopping splitter was StratifiedGroupKFold with seven folds, using the
first split with both classes in optimizer and stopping portions. The final
five members have their own frozen optimizer/stopping roles within development.
The public implementation **consumes these row roles directly** and verifies
coverage and group separation. It never rebuilds folds from anonymous names.

Training uses CPU, two torch threads, deterministic algorithms, batch size 64,
AdamW (learning rate 0.002, weight decay 0.0001), BCEWithLogitsLoss with
optimizer-only negative/positive class weighting, gradient clipping at 5,
at most 150 epochs, stopping AUC improvement greater than 1e-5, patience 25,
and restoration of the best state. There is no mixup, curve noise, weighted
sampler or learning-rate schedule. Five sigmoid member scores are cast to
float64 before averaging. These are uncalibrated growth scores.

The current paper's rule is score **>= 0.5**. This fixed value was adopted after
reviewing existing development/BALF results; the historical development-selected
0.49322897344827654 is not used. Model/representation selection also involved
previously examined BALF results. The BALF cohort is a retrospective feasibility
cohort, not untouched external validation. Its species context came from
subsequent pure-culture identification. Development grouped OOF performance and
retrospective BALF performance must be described with that history.

## MIC and complete denominators

For pure cultures, group by strain and full drug context. Technical repeats
must agree; both the observed and predicted no-drug controls must show growth.
After ordering measured exposed concentrations, all-negative series yield
`<=minimum`, growth at the highest tested level yields `>maximum`, and otherwise
the MIC is the next measured concentration after the highest growth call.
A single skipped non-growth concentration uses the higher endpoint and is
flagged; multiple skipped concentrations yield `NR`. No smoothing or label
changes are applied. Missing or discordant controls also yield `NR`.

The independent pure-culture comparator is the supplied `reference_mic` field,
whose experimental independence was confirmed by the author. The specific
reference method was not supplied, and original reports were not re-verified
for this release. Values, including `0.12`, are not rounded to a theoretical
dilution grid. Multiple different references within one series cause an error.
Only two exact numeric MICs enter the numeric agreement denominator. With
`d=log2(predicted/reference)`, exact agreement is `abs(d)<=1e-12`, and EA is
`abs(d)<=1+1e-12`. All 228 series remain: 194 reportable (82 numeric, 112 bounds)
and 34 NR. Of 82 numeric pairs, 51 agree exactly and 72 meet EA.

For BALF, classify each well at 0.5 **before** taking the majority of three
binary calls at each concentration. This is not thresholding the mean score.
Both observed-control and predicted-control majority must show growth. Apply
the same ordered concentration/skip endpoint rules to the resulting sequence.
Keep all eight specimens, including four NR specimens with negative observed
controls. Four reportable predictions from three participant groups are
compared with the recorded hospital/broth-recheck MICs. The comparator file
preserves the author-supplied correction and recheck provenance. Some source
reports were unavailable, and the reviewed image's exact specimen linkage
remained pending; this release does not claim all original reports or pairings
were independently re-verified. Units follow the project's µg/mL convention.

BALF uses the paper's frozen off-scale comparison from the
[CLSI BIT Part E instructions](https://clsi.org/media/i3ylyrde/part_e_accuracy_instructions_for_prefilled_excel_workbook.pdf)
and [Part F workbook](https://clsi.org/media/xh3jfjfn/part_f_ar_bank_data_entry_and_calculations.xlsm),
cells BIT-1 O7/Q7. The code retains the original complete formula branches;
reported `>` values use the next dilution only as an intermediate calculation.
Two `<=` results, or a known numeric result contained in the other reported
`<=` range, can have zero adjusted difference. EA is calculated by O7's
separate rule. Current comparisons are 1 vs 2, 1 vs <=2, and twice <=0.5 vs <=1:
EA is 4/4, and range-adjusted exact agreement is 3/4. A zero adjusted difference
does not establish identical unknown exact MICs. `NR` is never zero.

## Two reproduction modes and strict reconciliation

`saved` loads the fixed five member scores and ensemble scores, then recomputes
classification metrics, per-series MICs and complete denominators. It performs
zero model fits and works without torch. `train` fits all 25 OOF and five final
members, then performs the same calculations using the new predictions.

Before the first retraining, the released reconciliation policy was frozen:
maximum absolute difference <=1e-7 for every member and ensemble score;
no changed 0.5 well calls; every MIC series, qualifier and NR reason unchanged;
classification metric differences <=1e-12; and the 30 fits' stopping AUCs and
best/run epochs matching their reference records. On mismatch the command
retains all outputs, marks failure and exits nonzero. It never alters labels,
thresholds or roles to force agreement. It compares full series, not just EA.

Strict training requires Python 3.12.11, NumPy 2.3.5, pandas 2.3.3,
scikit-learn 1.6.1 and torch 2.10.0 (CPU build suffix allowed).
`requirements-paper.txt` pins these versions. Original training used macOS
arm64; other hardware/library builds may differ even with the same versions.
A failed numerical comparison is reported as a mismatch, not a reproduced
training result. Saved recomputation and full training have separate manifests.

Outputs go to new directories only. Each run records input/code/output hashes,
software/platform versions, mode, number of fits and reconciliation results.
Models, optimizer/stopping indices, scaling parameters and epoch histories are
saved locally by `train`. Pretrained checkpoints are not needed by `saved`.
All representation comparisons and eight-way ablations are outside this release.

## Teaching example and licenses

Use `glu-ntu-demo` explicitly for the preserved synthetic teaching workflow.
Its separate [data and methods](synthetic-data-and-methods.md) describe a
simplified model; its metrics are not paper results. The real workflow does
not use synthetic rows, generated labels or a synthetic threshold selection.

Code remains [MIT](../LICENSE). Real data have a separate
[CC BY 4.0 license and attribution notice](../LICENSE-DATA).
