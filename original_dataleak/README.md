# Original Data-Leak Paper Replica

This folder implements the paper-described KANInformer pipeline with the main suspected leakage condition:

```text
raw hourly data
-> paper preprocessing
-> season split
-> PCC report / paper seasonal inputs
-> VMD-CA-EWT decomposition on the full season
-> chronological train/validation/test split
-> train-only MinMax scaling
-> KANInformer-style model
-> h1/h2/h3 RMSE, MAE, MAPE
```

The intended leakage variable is:

```text
VMD-CA-EWT decomposition is done before train/test splitting.
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
