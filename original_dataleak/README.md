# Full-Season Decomposition Experiment

Read `REPRODUCTION_AUDIT.md` before describing this run as a paper reproduction.
The default implements the paper's written core architecture; exact published
metrics are not established, and the supplied autumn data is incomplete.

```powershell
python -m pip install -r requirements.txt
python checks.py
python run.py
```

`run.py` performs a reduced smoke run, removes its outputs, then starts the full run.
CUDA is selected automatically when available. All paths are relative to this file's
location; no fixed machine directory is required.

Flow: seasonal raw data -> full-season 3-sigma/spline cleaning -> computed PCC ->
VMD -> aggregate SE>0.4 components -> eight-mode Meyer EWT -> concatenate retained
IMFs, EWT modes, meteorology and WS -> chronological 8:1:1 split -> MinMax scaling ->
KANInformer -> h1/h2/h3 metrics and saved predictions/checkpoints.

This protocol intentionally permits future data to influence full-season processing.
The default scaler is train-only. `--scaler-scope full` inspects the public source's
additional normalization leakage.

`--architecture paper` uses the paper's Table 6 and generative decoder.
`--architecture author --e-layers 1` uses the public source's flattened KAN/recursive
prediction design. `--feature-selection paper` uses reported seasonal inputs instead
of enforcing the computed PCC rule. Every setting is saved in `results/summary.json`.

`--smoke-only` checks one reduced season and removes smoke outputs. Full outputs
remain in `results/`. Run the root `results.py` after both experiments finish.
