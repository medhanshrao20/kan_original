# No-Data-Leak Pipeline

This folder follows the same core paper architecture but makes it chronologically leakage-safe:

1. split each season into train/validation/test first,
2. fit cleaning statistics on train only,
3. select features using train only,
4. fit scaling on train only,
5. train on train, tune/check validation, and evaluate once on test.

The decomposition is also isolated by split so training features are not built from validation/test wind-speed values.

Run:

```powershell
python run.py
```

Fast smoke test only:

```powershell
python run.py --smoke-only
```

Outputs are written to `results/`.
