# Original Data-Leak Pipeline

This folder intentionally follows the paper-style order:

1. clean the full seasonal data,
2. select features on the full season,
3. decompose the full season using VMD-CA-EWT-style features,
4. scale the full season,
5. split into train/validation/test,
6. train the KANInformer-style model.

That means this pipeline is intentionally data-leaky. It is useful for testing the hypothesis that a paper-style full-series preprocessing flow can produce better-looking results.

Run:

```powershell
python run.py
```

Fast smoke test only:

```powershell
python run.py --smoke-only
```

Outputs are written to `results/`.
