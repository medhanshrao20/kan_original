# Paper-to-code audit

Paper: Leng et al., *Short-term wind speed forecasting based on a novel
KANInformer model and improved dual decomposition*, Energy (2025),
DOI 10.1016/j.energy.2025.135551.

The former standard-Transformer/RBF implementation was not an exact reproduction.
It has been replaced. This document records verified details and unresolved ones.
Do not describe this repository as a proven exact reproduction of the paper's numbers.

## Verified published choices

| Component | Paper location (PDF page) | Implemented choice |
|---|---|---|
| Station/period | Section 4.1, pp. 30-32 | Brentwood; Dec 2020-Feb 2021, Mar-May, Jun-Aug, Sep-Nov 2021 |
| Cleaning | Section 4.1, pp. 30-31 | 3-sigma flagging and cubic spline in original_dataleak |
| Input selection | Section 4.1, p. 33 | Compute PCC; absolute PCC >0.5 and p<0.05 |
| Reported inputs | Section 4.1, p. 33 | Optional --feature-selection paper uses the reported lists; computed results are still saved |
| Split | Section 4.1, p. 31 | Chronological 8:1:1, with forecast targets kept inside their own split |
| Scaling | Eq. (24), p. 31 | MinMaxScaler per feature; train fit by default |
| Decomposition timing | Section 3, Step 3, p. 29 | Full seasonal decomposition before partitioning in original_dataleak |
| VMD | Eqs. (1)-(6), pp. 14-16 | vmdpy ADMM VMD implementation, frequency-ordered components |
| Seasonal K | Table 4, p. 34 | Spring 11, summer 10, autumn 11, winter 10 |
| Component aggregation | Section 4.2/Table 5, pp. 34-35 | Sum all modes whose computed sample entropy exceeds 0.4 |
| EWT | Eqs. (8)-(16), pp. 17-18 | ewtpy empirical Meyer filter bank; analysis plus synthesis, not hard FFT bands |
| EWT count/window | Section 4.2, p. 34 | Eight EWT components; seven observed hourly inputs |
| KAN | Section 2.2.1, pp. 20-21 | Actual pykan cubic B-spline KAN; no RBF substitution |
| Informer | Section 2.2.2, pp. 22-26 | ProbSparse attention, convolution/ELU/pooling distillation, masked decoder, cross-attention |
| KAN placement | Section 2.2.3, p. 27 | Replaces feed-forward transformations inside every encoder/decoder block |
| Main dimensions | Table 6, p. 38 | d_model=64, heads=8, encoder layers=2, decoder layers=1, factor=5, batch=32, lr=0.001 |
| Metrics | Eqs. (26)-(28), p. 38 | RMSE, MAE, MAPE in percent; raw wind speed in m/s after inverse scaling |
| Reference results | Table 9, p. 45 | Published h1/h2/h3 values, kept separate from measured run metrics |

## Published source and paper conflict

Paper-linked repository: https://github.com/375330014/lzy.
Inspected commit: aebb26d9f46d2f3709ea176c51255d6955ed3707.
Unmodified source is retained as reference/author_KANInformer.py. It is not executed.

The source is a partially hardcoded experiment, not a complete reproduction package:

- It requires an unpublished, preprocessed input.csv and contains no VMD/CA/EWT code.
- Its default is one encoder layer; Table 6 states two.
- Its encoder KAN flattens all seven time steps and its decoder KAN flattens six.
  Section 2.2.3 describes nonlinear processing at each time step.
- It trains next-feature prediction and recursively appends forecasts for three steps.
  The Informer explanation in the paper describes zero-padded generative decoding.
- Its target slice -12:-8 assumes four features. It cannot work for an arbitrary
  dual-decomposition feature matrix without correcting target dimensions.
- It fits MinMaxScaler on the full supplied input before splitting supervised rows.
  The paper does not specify scaler fitting scope.
- It creates a tensor decoder mask but its attention implementation expects a mask
  object in some paths; ProbAttention constructs its own mask.
- It caches best_model_weights without copying tensors, does not enter eval mode
  during validation, and uses .numpy() directly on predictions that can be on CUDA.

There is no honest single implementation that is simultaneously identical to these
conflicting descriptions. Two execution options are explicit:

1. --architecture paper (default): time-step B-spline KAN transformations,
   Table 6 layer count, observed decoder start tokens plus zero future placeholders,
   three simultaneous wind-speed outputs. This follows the paper's written design.
2. --architecture author: flattened B-spline KAN transformations, shifted-history
   decoder, next-feature training, recursive h1/h2/h3 forecasts, and the published
   decoder mix layout. --e-layers 1 matches the source default; --e-layers 2 matches
   Table 6. Shape/device/checkpoint bugs are repaired. This is an adapted source path,
   not a byte-for-byte run of the authors' experiment.

## Details not supplied by the paper

These must not be described as recovered paper settings:

- VMD alpha, multiplier step/noise tolerance, initialization, stopping tolerance,
  and iteration limit. Current explicit defaults: alpha=2000, tau=0, tol=1e-7,
  uniform center initialization, and vmdpy's 500-iteration limit.
- Sample entropy embedding dimension, tolerance, and short-series convention.
  Current explicit defaults: m=2, radius=0.2 times standard deviation, common
  eligible templates, no self-matches. Zero match counts are reported as nonfinite.
- EWT peak detector, spectrum smoothing, boundary completion, and endpoint treatment.
  Current explicit choice: ewtpy locmax, average smoothing, completion enabled,
  package mirror extension. EWT modes use analysis and canonical dual synthesis.
  The finite-grid package bank has non-unit frame energy near Nyquist; dividing
  synthesis by the summed squared filter response restores reconstruction.
  This correction is explicit; the authors' finite-grid EWT implementation is unknown.
- Cubic spline boundary conditions, extrapolation, exact outlier-fitting population,
  unit conversion/rounding, and whether zeros receive special treatment.
  Current original path: per-season 3-sigma, not-a-knot spline, spline extrapolation.
- The decoder start-token length in the hybrid experiment. Current paper-mode
  default: four observed tokens; configurable through --label-len.
- Original random seeds, package versions for pykan/VMD/EWT, saved weights,
  dropout for the final hybrid model, and exact selection of the best training epoch.
  KAN grid=5, degree=3 and hidden width=20 are from public source; epochs=200,
  Adam, early-stopping patience=3 and dropout=0 are source-derived choices.
- Whether splits occur before or after constructing supervised windows.
  Current code splits raw chronological positions and excludes targets that cross
  train/validation boundaries. Source partitions already-constructed windows.
- Quantitative rule for "no significant downward trend" in residual energy ratio.
  Reported seasonal K is used directly; Table 4 is not claimed to be regenerated.

## Dataset mismatch

The supplied CSV has 8064 rows, station 47, Brentwood. Its raw date range is
2020-12-01 0100 through 2021-11-01 2400. Hour 2400 maps to the following midnight.
Autumn contains only 1489 observations under this convention, versus 2184 expected
for September-November. Winter has 2159 versus 2160 because the first midnight
is absent. Missing seasonal tails are not synthesized. --require-paper-data stops
when seasonal coverage is incomplete. This CSV cannot reproduce all four exact
experimental datasets described in the paper.

## Reliable comparison path

no_dataleak fits outlier statistics and PCC on training rows only, imputes using
past values with a training-median fallback, and fits scaling on training inputs.
VMD/CA/EWT are recomputed on a history ending before each forecast target.
Training fixes the set of high-entropy mode identities. At validation/test time,
neither target wind speed nor later observed weather enters the input matrix.
Later test observations may enter subsequent forecasts once they have been observed.

This is an operationally causal adaptation. It changes cleaning, feature-selection
scope and decomposition timing, so comparing it to the original path does not
isolate decomposition as the only intervention. A split-first decomposition of an
entire test segment would still expose each input to future test observations.
Both default to train-only scaling. --scaler-scope full is available only in
original_dataleak to inspect the additional scaling leakage in public source.

## Scope still not reproduced

This entry point covers the proposed predictor and its two evaluation protocols.
It does not regenerate every experiment in the paper: Table 7's nine baselines,
EMD/EEMD/CEEMDAN ablations, VMD-only/VMD-EWT comparisons, Table 10's four external
hybrid models, all grid/random-search and five-fold runs, or the published SHAP
figures. Those experiments have additional implementation choices that are not
fully published. They must not be reported as implemented or reproduced by this run.

Exact published scores are an empirical claim requiring matched data/settings and
full training, not a guarantee implied by an architecture or a passing smoke test.

## Verification on the supplied CSV

On October 5, 2026, the four full supplied seasonal signals completed the corrected
VMD/CA/EWT calculation with their reported K values. EWT aggregate reconstruction
errors were below 4e-15. Spring's computed PCC selection was WS/RH instead of the
paper's WS/ET; summer, autumn and winter matched the reported variable lists.
Computed high-entropy IMF identities also differed from Table 5. These are measured
differences, not evidence that Table 5 was reproduced. Changing undocumented
parameters until they resemble that table would not establish the original settings.

Full-size CPU forward/backward checks passed for both architecture modes, including
recursive h1/h2/h3 forecasts. The future-suffix perturbation check, sample-attention
equation comparison, odd-length handling and EWT reconstruction checks passed.
Reduced training/evaluation smoke runs passed. No full-training score or CUDA
verification is claimed by these checks. Normal entry points execute method checks
and a reduced smoke run before full training; --skip-smoke explicitly bypasses them.
