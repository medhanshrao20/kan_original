# No-Data-Leak Paper Replica

This folder implements the same paper-described KANInformer pipeline, but moves the suspected leakage step after the chronological split:

```text
raw hourly data
-> paper preprocessing
-> season split
-> PCC report / paper seasonal inputs
-> chronological train/validation/test split
-> VMD-CA-EWT decomposition separately on train, validation, and test
-> train-only MinMax scaling
-> KANInformer-style model
-> h1/h2/h3 RMSE, MAE, MAPE
```

The intended controlled difference from `original_dataleak` is:

```text
VMD-CA-EWT decomposition is done after train/test splitting.
```

Run:

```powershell
python run.py
```

Fast smoke test:

```powershell
python run.py --smoke-only
```

Results are written to `results/`.
