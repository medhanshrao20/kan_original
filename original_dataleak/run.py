from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import shutil
import time
import unittest
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import CubicSpline
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from decomposition import decompose
from model import KANInformer

PIPELINE_NAME = "original_dataleak"
DECOMPOSE_BEFORE_SPLIT = True
ROOT = Path(__file__).resolve().parent
VARIABLES = ["ET", "PCP", "SR", "VP", "AT", "RH", "DPT", "WS", "WD", "ST"]
SEASONS = ["spring", "summer", "autumn", "winter"]
PERIODS = {"spring": ("2021-03-01", "2021-06-01"), "summer": ("2021-06-01", "2021-09-01"),
           "autumn": ("2021-09-01", "2021-12-01"), "winter": ("2020-12-01", "2021-03-01")}
PAPER_INPUTS = {"spring": ["WS", "ET"], "summer": ["WS", "ET", "AT", "RH"],
                "autumn": ["WS", "ET"], "winter": ["WS"]}
PAPER_K = {"spring": 11, "summer": 10, "autumn": 11, "winter": 10}
PAPER_METRICS = {
    "spring": [(0.071, 0.056, 2.0), (0.100, 0.078, 2.7), (0.112, 0.090, 3.3)],
    "summer": [(0.064, 0.051, 3.1), (0.086, 0.066, 3.9), (0.115, 0.091, 5.8)],
    "autumn": [(0.061, 0.048, 3.8), (0.078, 0.063, 5.2), (0.084, 0.066, 5.4)],
    "winter": [(0.119, 0.089, 4.5), (0.160, 0.125, 6.3), (0.192, 0.148, 7.8)],
}


@dataclass
class Config:
    architecture: str = "paper"
    epochs: int = 200
    batch_size: int = 32
    lr: float = 0.001
    d_model: int = 64
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    factor: int = 5
    dropout: float = 0.0
    window: int = 7
    horizons: int = 3
    label_len: int = 4
    kan_hidden: int = 20
    kan_grid: int = 5
    patience: int = 3
    seed: int = 42
    vmd_alpha: float = 2000.0
    vmd_tau: float = 0.0
    vmd_tol: float = 1e-7
    se_m: int = 2
    se_r: float = 0.2
    se_threshold: float = 0.4
    ewt_modes: int = 8
    decomposition_context: int = 0
    min_context: int = 64
    feature_selection: str = "pcc"
    scaler_scope: str = "train"
    smoke: bool = False


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_data(path):
    raw = pd.read_csv(path)
    if raw["Stn Id"].nunique() != 1 or raw["Stn Name"].iloc[0] != "Brentwood":
        raise ValueError("Expected the single Brentwood station used in the paper.")
    hours = raw["Hour (PST)"].astype(int)
    timestamp = pd.to_datetime(raw["Date"], format="%m/%d/%Y") + pd.to_timedelta(hours // 100, unit="h")
    columns = ["ETo (in)", "Precip (in)", "Sol Rad (Ly/day)", "Vap Pres (mBars)", "Air Temp (F)",
               "Rel Hum (%)", "Dew Point (F)", "Wind Speed (mph)", "Wind Dir (0-360)", "Soil Temp (F)"]
    frame = pd.DataFrame({name: pd.to_numeric(raw[col], errors="coerce").to_numpy()
                          for name, col in zip(VARIABLES, columns)}, index=pd.DatetimeIndex(timestamp))
    frame.index.name = "timestamp"
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    if frame.index.has_duplicates:
        raise ValueError("Duplicate hourly timestamps: resolve explicitly before training.")
    frame.sort_index(inplace=True)
    for col in ["AT", "DPT", "ST"]:
        frame[col] = (frame[col] - 32) * 5 / 9
    frame["WS"] *= 0.44704
    return frame.reindex(pd.date_range(frame.index.min(), frame.index.max(), freq="h", name="timestamp"))


def clean_data(frame, train_end, causal):
    fit = frame.iloc[:train_end] if causal else frame
    mean, std = fit.mean(), fit.std(ddof=1)
    mask = (frame - mean).abs() > 3 * std
    result = frame.mask(mask)
    if causal:
        result = result.ffill().fillna(fit.mask((fit - mean).abs() > 3 * std).median())
    else:
        coordinates = np.arange(len(frame))
        for col in VARIABLES:
            valid = result[col].notna().to_numpy()
            if valid.sum() < 4:
                raise ValueError(f"Insufficient valid values for cubic interpolation: {col}")
            spline = CubicSpline(coordinates[valid], result.loc[valid, col].to_numpy(), bc_type="not-a-knot")
            result.loc[~valid, col] = spline(coordinates[~valid])
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Cleaning left nonfinite data.")
    return result, {"missing_before": frame.isna().sum().to_dict(), "outliers": mask.sum().to_dict(),
                    "fit_scope": "training only" if causal else "full season",
                    "imputation": "forward fill, training median fallback" if causal else "cubic spline, not-a-knot"}


def select_features(frame):
    rows, selected = [], ["WS"]
    for col in VARIABLES:
        if col == "WS":
            continue
        if frame[col].std() == 0 or frame["WS"].std() == 0:
            r, p = None, None
        else:
            r, p = (float(v) for v in pearsonr(frame[col], frame["WS"]))
        take = r is not None and abs(r) > 0.5 and p < 0.05
        rows.append({"feature": col, "pcc": r, "p_value": p, "selected": take})
        if take:
            selected.append(col)
    return selected, rows


def feature_matrix(frame, selected, components):
    meteo = [col for col in selected if col != "WS"]
    return np.column_stack([frame[meteo].to_numpy(), components, frame[["WS"]].to_numpy()])


def split_masks(origins, config, train_end, val_end):
    return [(origins + config.horizons <= train_end),
            (origins >= train_end) & (origins + config.horizons <= val_end), (origins >= val_end)]


def full_season_windows(frame, selected, k, config, train_end, val_end):
    components, report = decompose(frame["WS"].to_numpy(), k, config)
    features = feature_matrix(frame, selected, components)
    scaler = MinMaxScaler().fit(features if config.scaler_scope == "full" else features[:train_end])
    scaled = scaler.transform(features)
    windows = np.lib.stride_tricks.sliding_window_view(scaled, config.window, axis=0).transpose(0, 2, 1)
    origins = np.arange(config.window, len(frame) - config.horizons + 1)
    xs = windows[origins - config.window].astype(np.float32)
    raw_targets = frame["WS"].to_numpy()[origins[:, None] + np.arange(config.horizons)]
    ys = scaled[origins] if config.architecture == "author" else raw_targets * scaler.scale_[-1] + scaler.min_[-1]
    masks = split_masks(origins, config, train_end, val_end)
    blocks = [(xs[m], ys[m].astype(np.float32), raw_targets[m], origins[m]) for m in masks]
    labels = [col for col in selected if col != "WS"] + report["labels"] + ["WS"]
    return blocks, scaler, report, labels


def causal_windows(frame, selected, k, config, train_end, val_end):
    _, train_report = decompose(frame["WS"].iloc[:train_end].to_numpy(), k, config)
    high = train_report["high_indices"]
    labels = [col for col in selected if col != "WS"] + train_report["labels"] + ["WS"]
    origins = np.arange(max(config.window, config.min_context), len(frame) - config.horizons + 1)
    matrices = np.empty((len(origins), config.window, len(labels)), dtype=np.float32)
    raw_targets = frame["WS"].to_numpy()[origins[:, None] + np.arange(config.horizons)]
    feature_targets = np.zeros((len(origins), len(labels)), dtype=np.float32)
    last_report = None
    for i, origin in enumerate(origins):
        start = max(0, origin - config.decomposition_context) if config.decomposition_context else 0
        history = frame.iloc[start:origin]
        components, last_report = decompose(history["WS"].to_numpy(), k, config, fixed_high=high)
        matrices[i] = feature_matrix(history, selected, components)[-config.window:]
        if config.architecture == "author":
            target_history = frame.iloc[start:origin + 1]
            target_parts, _ = decompose(target_history["WS"].to_numpy(), k, config, fixed_high=high)
            feature_targets[i] = feature_matrix(target_history, selected, target_parts)[-1]
        if i == 0 or (i + 1) % 50 == 0 or i == len(origins) - 1:
            print(f"    causal decomposition {i + 1}/{len(origins)}, last observation={frame.index[origin - 1]}", flush=True)
    masks = split_masks(origins, config, train_end, val_end)
    scaler = MinMaxScaler().fit(matrices[masks[0]].reshape(-1, len(labels)))
    xs = scaler.transform(matrices.reshape(-1, len(labels))).reshape(matrices.shape).astype(np.float32)
    ys = scaler.transform(feature_targets) if config.architecture == "author" else raw_targets * scaler.scale_[-1] + scaler.min_[-1]
    report = {"training_mode_selection": train_report, "last_origin": last_report, "causal": True,
              "decomposition_context": config.decomposition_context,
              "discarded_boundary_windows": int(sum(~(masks[0] | masks[1] | masks[2])))}
    return [(xs[m], ys[m].astype(np.float32), raw_targets[m], origins[m]) for m in masks], scaler, report, labels


def metrics(actual, predicted):
    diff = predicted - actual
    if np.any(actual == 0):
        raise ValueError("MAPE Eq. (28) undefined at zero wind speed; no silent epsilon substitution.")
    return {"rmse": float(np.sqrt(np.mean(diff ** 2))), "mae": float(np.mean(np.abs(diff))),
            "mape": float(100 * np.mean(np.abs(diff / actual)))}


def train(blocks, config, device, out, season, scaler):
    train_x, train_y, _, _ = blocks[0]
    val_x, val_y, _, _ = blocks[1]
    if not len(train_x) or not len(val_x) or not len(blocks[2][0]):
        raise ValueError("Empty split after constructing windows.")
    seed_all(config.seed)
    model = KANInformer(train_x.shape[-1], config, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
                        batch_size=config.batch_size, shuffle=False, pin_memory=device.type == "cuda")
    validation = DataLoader(TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)), batch_size=config.batch_size)
    best, best_weights, stale, history = float("inf"), None, 0, []
    print(f"    model={config.architecture}, parameters={sum(p.numel() for p in model.parameters()):,}; windows train/val/test={[len(b[0]) for b in blocks]}", flush=True)
    for epoch in range(config.epochs):
        model.train()
        loss_total = torch.zeros((), device=device)
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss.")
            loss.backward()
            optimizer.step()
            loss_total += loss.detach() * len(x)
            if step == 0 or (step + 1) % 20 == 0:
                print(f"    epoch {epoch + 1}/{config.epochs}, batch {step + 1}/{len(loader)}, loss={loss.detach().item():.6f}", flush=True)
        model.eval()
        val_total = torch.zeros((), device=device)
        with torch.no_grad():
            for x, y in validation:
                x, y = x.to(device), y.to(device)
                val_total += torch.nn.functional.mse_loss(model(x), y) * len(x)
        val_loss = val_total.item() / len(val_x)
        history.append({"epoch": epoch + 1, "train_mse": loss_total.item() / len(train_x), "val_mse": val_loss})
        print(f"    epoch {epoch + 1}: train={history[-1]['train_mse']:.6f}, val={val_loss:.6f}", flush=True)
        if val_loss < best:
            best, stale = val_loss, 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= config.patience:
                print("    early stopping; restoring best validation checkpoint", flush=True)
                break
    model.load_state_dict(best_weights)
    torch.save({"model_state": best_weights, "config": asdict(config), "scaler_scale": scaler.scale_.tolist(),
                "scaler_offset": scaler.min_.tolist()}, out / f"{season}_checkpoint.pt")
    pd.DataFrame(history).to_csv(out / f"{season}_training.csv", index=False)
    test_x, _, actual, origins = blocks[2]
    predicted = np.empty_like(actual)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(test_x), config.batch_size):
            x = torch.from_numpy(test_x[start:start + config.batch_size]).to(device)
            prediction = model.forecast(x).cpu().numpy()
            predicted[start:start + len(x)] = (prediction - scaler.min_[-1]) / scaler.scale_[-1]
    records = []
    for horizon in range(config.horizons):
        records.extend({"origin_index": int(origin), "horizon": horizon + 1, "actual": float(a), "prediction": float(p)}
                       for origin, a, p in zip(origins, actual[:, horizon], predicted[:, horizon]))
    pd.DataFrame(records).to_csv(out / f"{season}_predictions.csv", index=False)
    return [{"pipeline": PIPELINE_NAME, "season": season, "horizon": h + 1,
             **metrics(actual[:, h], predicted[:, h])} for h in range(config.horizons)]


def run(config, path, out, seasons, require_complete=False):
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Pipeline: {PIPELINE_NAME}; CUDA available: {torch.cuda.is_available()}; device: {device}", flush=True)
    print("Execution of published methods; exact numerical reproduction is unverified. Read REPRODUCTION_AUDIT.md.", flush=True)
    raw = load_data(path)
    summaries, all_metrics = {}, []
    for season in seasons:
        start, end = PERIODS[season]
        frame = raw.loc[(raw.index >= start) & (raw.index < end)].copy()
        expected = len(pd.date_range(start, end, freq="h", inclusive="left"))
        complete = len(frame) == expected and not frame.isna().all(axis=1).any()
        print(f"\n{season}: supplied {len(frame)}/{expected} expected hourly rows; complete={complete}", flush=True)
        if require_complete and not complete:
            raise ValueError(f"{season}: supplied CSV does not cover the paper season. No synthetic tail will be created.")
        if config.smoke:
            frame = frame.iloc[:240]
        train_end, val_end = int(len(frame) * 0.8), int(len(frame) * 0.9)
        frame, cleaning = clean_data(frame, train_end, causal=not DECOMPOSE_BEFORE_SPLIT)
        pcc_frame = frame if DECOMPOSE_BEFORE_SPLIT else frame.iloc[:train_end]
        selected, pcc = select_features(pcc_frame)
        computed = list(selected)
        if config.feature_selection == "paper":
            selected = PAPER_INPUTS[season]
        print(f"    computed PCC inputs={computed}; used={selected}; reported paper inputs={PAPER_INPUTS[season]}", flush=True)
        k = 3 if config.smoke else PAPER_K[season]
        begin = time.perf_counter()
        function = full_season_windows if DECOMPOSE_BEFORE_SPLIT else causal_windows
        blocks, scaler, decomposition, labels = function(frame, selected, k, config, train_end, val_end)
        print(f"    decomposition complete in {time.perf_counter() - begin:.1f}s; features={labels}", flush=True)
        rows = train(blocks, config, device, out, season, scaler)
        all_metrics.extend(rows)
        pd.DataFrame(pcc).to_csv(out / f"{season}_pcc.csv", index=False)
        summaries[season] = {"complete_paper_season": complete, "expected_hours": expected, "rows_used": len(frame),
                             "cleaning": cleaning, "selected_inputs": selected, "computed_inputs": computed,
                             "paper_inputs_match": set(computed) == set(PAPER_INPUTS[season]),
                             "decomposition": decomposition, "feature_columns": labels}
        for row in rows:
            print(f"    h{row['horizon']}: RMSE={row['rmse']:.6f}, MAE={row['mae']:.6f}, MAPE={row['mape']:.3f}%", flush=True)
        pd.DataFrame(all_metrics).to_csv(out / "metrics.csv", index=False)
    payload = {"pipeline": PIPELINE_NAME, "config": asdict(config), "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "exact_replication": False, "decompose_before_split": DECOMPOSE_BEFORE_SPLIT,
               "reports": summaries, "metrics": all_metrics, "torch_version": torch.__version__, "device": str(device)}
    (out / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["| Season | Horizon | Paper RMSE | Run RMSE | Paper MAE | Run MAE | Paper MAPE | Run MAPE |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in all_metrics:
        p = PAPER_METRICS[row["season"]][row["horizon"] - 1]
        lines.append(f"| {row['season']} | h{row['horizon']} | {p[0]:.3f} | {row['rmse']:.6f} | {p[1]:.3f} | {row['mae']:.6f} | {p[2]:.1f}% | {row['mape']:.3f}% |")
    (out / "paper_format_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run documented KANInformer methods; numerical equivalence is not established.")
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/hourly.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--seasons", nargs="+", choices=SEASONS, default=SEASONS)
    parser.add_argument("--require-paper-data", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    defaults = Config()
    for name, value in asdict(defaults).items():
        if name != "smoke":
            choices = {"architecture": ["paper", "author"], "feature_selection": ["pcc", "paper"], "scaler_scope": ["train", "full"]}.get(name)
            parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value, choices=choices)
    args = parser.parse_args()
    config = Config(**{name: getattr(args, name) for name in asdict(defaults) if name != "smoke"})
    if config.d_model % config.n_heads or config.d_model % 2:
        parser.error("d-model must be even and divisible by n-heads")
    if not 1 <= config.label_len <= config.window:
        parser.error("label-len must be within the input window")
    if config.epochs < 1 or config.batch_size < 1 or config.min_context < 16:
        parser.error("epochs/batch-size must be positive and min-context must be at least 16")
    if config.decomposition_context and config.decomposition_context < config.min_context:
        parser.error("decomposition-context must be zero or at least min-context")
    if not DECOMPOSE_BEFORE_SPLIT and config.scaler_scope != "train":
        parser.error("no_dataleak requires training-only scaler fitting")
    if not args.skip_smoke or args.smoke_only:
        from checks import MethodChecks
        threads = torch.get_num_threads()
        try:
            verified = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(MethodChecks))
        finally:
            torch.set_num_threads(threads)
        if not verified.wasSuccessful():
            raise RuntimeError("Method checks failed; full training will not start.")
        smoke = replace(config, smoke=True, epochs=1, d_model=8, n_heads=2, kan_hidden=4, ewt_modes=3,
                        min_context=32, decomposition_context=64, vmd_tol=1e-4)
        smoke_dir = args.output / "_smoke"
        if smoke_dir.exists():
            parser.error(f"Existing smoke outputs at {smoke_dir}; choose another --output")
        run(smoke, args.data, smoke_dir, [args.seasons[0]])
        resolved = smoke_dir.resolve()
        if resolved.parent != args.output.resolve() or resolved.name != "_smoke":
            raise RuntimeError("Unsafe smoke cleanup path")
        shutil.rmtree(resolved)
        print("Smoke passed; smoke artifacts removed.", flush=True)
        if args.smoke_only:
            return
    run(config, args.data, args.output, args.seasons, args.require_paper_data)


if __name__ == "__main__":
    main()
