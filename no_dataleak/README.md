# Causal KANInformer Experiment

Read `REPRODUCTION_AUDIT.md` for differences from the published experiment.
This folder is self-contained and uses the same neural/decomposition implementations
as original_dataleak.

```powershell
python -m pip install -r requirements.txt
python checks.py
python run.py
```

The normal entry point smoke-tests first, deletes its own smoke artifacts, then
trains. CUDA is selected automatically when available; decomposition runs on CPU.

Flow: chronological seasonal partition -> fit outlier/PCC rules on training data ->
causal forward imputation -> decompose each observed history with VMD -> aggregate
training-selected high-entropy mode identities -> empirical Meyer EWT -> input
windows ending before targets -> training-only MinMax scaling -> KANInformer ->
h1/h2/h3 metrics and saved predictions/checkpoints.

Validation/test targets and future meteorology never enter forecast inputs. Entire
test blocks are not decomposed ahead of forecasting. This changes cleaning and PCC
scope as well as decomposition timing, so it is not a single-variable leakage test.

Use `--decomposition-context 256` to cap observed history if preprocessing is slow.
The default is all observed history; changing it is a recorded experimental choice.
`--architecture paper` is the default. `--architecture author` provides the adapted
public-source flattened-KAN/recursive mode. Full-season scaler fitting is rejected.

`--smoke-only` checks a reduced season and removes smoke outputs. Full results are
stored in this folder's `results/`. Then run `python ../results.py` after both folders
have produced matching seasonal results.
