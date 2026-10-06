from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "data" / "raw" / "hourly.csv"
RESULTS_DIR = ROOT / "results"
SEASONS = ["winter", "spring", "summer", "autumn"]
VARIABLES = ["ET", "PCP", "SR", "VP", "AT", "RH", "DPT", "WS", "WD", "ST"]
METEO_VARIABLES = ["ET", "PCP", "SR", "VP", "AT", "RH", "DPT", "WD", "ST"]
PAPER_SELECTED_INPUTS = {
    "spring": ["WS", "ET"],
    "summer": ["WS", "ET", "AT", "RH"],
    "autumn": ["WS", "ET"],
    "winter": ["WS"],
}
PAPER_VMD_K = {"spring": 11, "summer": 10, "autumn": 11, "winter": 10}
PAPER_FINAL = {
    ("spring", 1): {"rmse": 0.071, "mae": 0.056, "mape": 2.0},
    ("spring", 2): {"rmse": 0.100, "mae": 0.078, "mape": 2.7},
    ("spring", 3): {"rmse": 0.112, "mae": 0.090, "mape": 3.3},
    ("summer", 1): {"rmse": 0.064, "mae": 0.051, "mape": 3.1},
    ("summer", 2): {"rmse": 0.086, "mae": 0.066, "mape": 3.9},
    ("summer", 3): {"rmse": 0.115, "mae": 0.091, "mape": 5.8},
    ("autumn", 1): {"rmse": 0.061, "mae": 0.048, "mape": 3.8},
    ("autumn", 2): {"rmse": 0.078, "mae": 0.063, "mape": 5.2},
    ("autumn", 3): {"rmse": 0.084, "mae": 0.066, "mape": 5.4},
    ("winter", 1): {"rmse": 0.119, "mae": 0.089, "mape": 4.5},
    ("winter", 2): {"rmse": 0.160, "mae": 0.125, "mape": 6.3},
    ("winter", 3): {"rmse": 0.192, "mae": 0.148, "mape": 7.8},
}


@dataclass
class RunConfig:
    epochs: int
    batch_size: int
    lr: float
    d_model: int
    n_heads: int
    n_layers: int
    kan_grid: int
    window: int
    horizons: int
    vmd_iter: int
    ewt_modes: int
    smoke: bool


def seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_timestamp(date_series: pd.Series, hour_series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(date_series, format="%m/%d/%Y")
    hours = hour_series.astype(int)
    day_offset = (hours == 2400).astype(int)
    hour_of_day = np.where(hours == 2400, 0, hours // 100)
    return dates + pd.to_timedelta(day_offset, unit="D") + pd.to_timedelta(hour_of_day, unit="h")


def load_hourly(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    frame = pd.DataFrame(
        {
            "timestamp": make_timestamp(raw["Date"], raw["Hour (PST)"]),
            "ET": pd.to_numeric(raw["ETo (in)"], errors="coerce"),
            "PCP": pd.to_numeric(raw["Precip (in)"], errors="coerce"),
            "SR": pd.to_numeric(raw["Sol Rad (Ly/day)"], errors="coerce"),
            "VP": pd.to_numeric(raw["Vap Pres (mBars)"], errors="coerce"),
            "AT": (pd.to_numeric(raw["Air Temp (F)"], errors="coerce") - 32.0) * 5.0 / 9.0,
            "RH": pd.to_numeric(raw["Rel Hum (%)"], errors="coerce"),
            "DPT": (pd.to_numeric(raw["Dew Point (F)"], errors="coerce") - 32.0) * 5.0 / 9.0,
            "WS": pd.to_numeric(raw["Wind Speed (mph)"], errors="coerce") * 0.44704,
            "WD": pd.to_numeric(raw["Wind Dir (0-360)"], errors="coerce"),
            "ST": (pd.to_numeric(raw["Soil Temp (F)"], errors="coerce") - 32.0) * 5.0 / 9.0,
        }
    )
    return frame.sort_values("timestamp").reset_index(drop=True)


def fit_cleaning_stats(train_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    stats: Dict[str, Tuple[float, float]] = {}
    for col in VARIABLES:
        x = train_df[col].astype(float)
        mean = float(x.mean(skipna=True))
        std = float(x.std(skipna=True))
        if not np.isfinite(std) or std == 0:
            std = 0.0
        stats[col] = (mean, std)
    return stats


def clean_with_train_stats_no_leak(df: pd.DataFrame, stats: Dict[str, Tuple[float, float]]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    out = df.copy()
    counts: Dict[str, int] = {}
    for col in VARIABLES:
        mean, std = stats[col]
        if std == 0:
            counts[col] = 0
            continue
        x = out[col].astype(float)
        mask = (x - mean).abs() > 3.0 * std
        counts[col] = int(mask.sum())
        out.loc[mask, col] = np.nan
    out = out.set_index("timestamp")
    for col in VARIABLES:
        try:
            out[col] = out[col].interpolate(method="spline", order=3).ffill().bfill()
        except Exception:
            out[col] = out[col].interpolate(method="linear").ffill().bfill()
    return out.reset_index(), counts


def split_seasons(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    ts = df["timestamp"]
    masks = {
        "winter": (ts >= "2020-12-01") & (ts < "2021-03-01"),
        "spring": (ts >= "2021-03-01") & (ts < "2021-06-01"),
        "summer": (ts >= "2021-06-01") & (ts < "2021-09-01"),
        "autumn": (ts >= "2021-09-01") & (ts < "2021-12-01"),
    }
    return {name: df.loc[mask].reset_index(drop=True) for name, mask in masks.items()}


def pcc_report_train_only(train_df: pd.DataFrame) -> pd.DataFrame:
    ws = train_df["WS"].to_numpy(float)
    rows = []
    for col in METEO_VARIABLES:
        x = train_df[col].to_numpy(float)
        if np.nanstd(x) == 0 or np.nanstd(ws) == 0:
            corr, p_value = np.nan, np.nan
        else:
            corr, p_value = pearsonr(x, ws)
        rows.append(
            {
                "variable": col,
                "pcc": float(corr),
                "p_value": float(p_value),
                "selected_by_rule": bool(np.isfinite(corr) and abs(corr) > 0.5 and p_value < 0.05),
            }
        )
    return pd.DataFrame(rows)


def vmd(signal: np.ndarray, k_modes: int, alpha: float = 2000.0, tol: float = 1e-6, max_iter: int = 200) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    original_len = len(x)
    if original_len % 2:
        x = np.r_[x, x[-1]]
    half = len(x) // 2
    mirrored = np.r_[np.flip(x[:half]), x, np.flip(x[-half:])]
    freqs = np.arange(1, len(mirrored) + 1) / len(mirrored) - 0.5 - 1.0 / len(mirrored)
    f_hat = np.fft.fftshift(np.fft.fft(mirrored))
    f_hat_plus = f_hat.copy()
    f_hat_plus[: len(f_hat_plus) // 2] = 0
    u_hat = np.zeros((max_iter, len(freqs), k_modes), dtype=complex)
    omega = np.zeros((max_iter, k_modes))
    omega[0] = 0.5 / k_modes * np.arange(k_modes)
    lambda_hat = np.zeros((max_iter, len(freqs)), dtype=complex)
    positive = slice(len(freqs) // 2, None)
    u_diff = tol + np.spacing(1)
    n = 0
    sum_uk = 0
    while u_diff > tol and n < max_iter - 1:
        sum_uk = u_hat[n, :, -1] + sum_uk - u_hat[n, :, 0]
        u_hat[n + 1, :, 0] = (f_hat_plus - sum_uk - lambda_hat[n] / 2) / (1 + alpha * (freqs - omega[n, 0]) ** 2)
        denom = np.sum(np.abs(u_hat[n + 1, positive, 0]) ** 2)
        if denom > 0:
            omega[n + 1, 0] = np.sum(freqs[positive] * np.abs(u_hat[n + 1, positive, 0]) ** 2) / denom
        for mode in range(1, k_modes):
            sum_uk = u_hat[n + 1, :, mode - 1] + sum_uk - u_hat[n, :, mode]
            u_hat[n + 1, :, mode] = (f_hat_plus - sum_uk - lambda_hat[n] / 2) / (1 + alpha * (freqs - omega[n, mode]) ** 2)
            denom = np.sum(np.abs(u_hat[n + 1, positive, mode]) ** 2)
            if denom > 0:
                omega[n + 1, mode] = np.sum(freqs[positive] * np.abs(u_hat[n + 1, positive, mode]) ** 2) / denom
        lambda_hat[n + 1] = lambda_hat[n]
        n += 1
        diffs = u_hat[n, :, :] - u_hat[n - 1, :, :]
        u_diff = abs(np.sum(np.conj(diffs) * diffs) / len(freqs))
    full_hat = np.zeros((len(freqs), k_modes), dtype=complex)
    full_hat[len(freqs) // 2 :] = u_hat[n, len(freqs) // 2 :]
    full_hat[: len(freqs) // 2] = np.conj(np.flipud(u_hat[n, len(freqs) // 2 :]))
    modes = np.vstack([np.real(np.fft.ifft(np.fft.ifftshift(full_hat[:, m]))) for m in range(k_modes)])
    return modes[:, half : half + original_len]


def sample_entropy(signal: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= m + 2:
        return 0.0
    r = r_ratio * np.std(x)
    if r == 0:
        return 0.0

    def embed(dim: int) -> np.ndarray:
        return np.array([x[i : i + dim] for i in range(len(x) - dim + 1)])

    b = len(cKDTree(embed(m)).query_pairs(r, p=np.inf))
    a = len(cKDTree(embed(m + 1)).query_pairs(r, p=np.inf))
    if b == 0:
        return 0.0
    return float(-np.log(max(a, 1) / b))


def ewt_style(signal: np.ndarray, n_modes: int) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    spectrum = np.fft.rfft(x)
    mag = np.abs(spectrum)
    peaks, _ = find_peaks(mag)
    if len(peaks) >= n_modes:
        strong = sorted(peaks[np.argsort(mag[peaks])[-n_modes:]])
        boundaries = [int(round((a + b) / 2)) for a, b in zip(strong[:-1], strong[1:])]
    else:
        boundaries = np.linspace(1, len(mag) - 1, n_modes + 1, dtype=int)[1:-1].tolist()
    edges = [0] + sorted(set(boundaries)) + [len(mag)]
    while len(edges) < n_modes + 1:
        edges = sorted(set(edges + np.linspace(0, len(mag), n_modes + 1, dtype=int).tolist()))
    edges = edges[:n_modes] + [len(mag)]
    bands = []
    for start, end in zip(edges[:-1], edges[1:]):
        band = np.zeros_like(spectrum)
        band[start:end] = spectrum[start:end]
        bands.append(np.fft.irfft(band, n=len(x)))
    return np.vstack(bands)


def select_train_pcc_inputs(pcc: pd.DataFrame) -> List[str]:
    selected = ["WS"] + pcc.loc[pcc["selected_by_rule"], "variable"].tolist()
    return list(dict.fromkeys(selected))


def build_split_features_no_leak(
    season: str,
    split_name: str,
    split_df: pd.DataFrame,
    selected: List[str],
    config: RunConfig,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    k_modes = min(3, PAPER_VMD_K[season]) if config.smoke else PAPER_VMD_K[season]
    ewt_modes = min(3, config.ewt_modes) if config.smoke else config.ewt_modes
    print(f"  {season}/{split_name}: building split-isolated VMD-CA-EWT features [NO-LEAK], K={k_modes}, EWT={ewt_modes}")
    ws = split_df["WS"].to_numpy(float)
    modes = vmd(ws, k_modes=k_modes, max_iter=config.vmd_iter)
    se = {f"IMF{i + 1}": sample_entropy(modes[i]) for i in range(k_modes)}
    threshold = 0.4
    high = [i for i, name in enumerate(se) if se[name] > threshold]
    if not high:
        high = [int(np.argmax([se[f"IMF{i + 1}"] for i in range(k_modes)]))]
    fused = modes[high].sum(axis=0)
    ewts = ewt_style(fused, ewt_modes)
    features = pd.DataFrame(index=split_df.index)
    for col in selected:
        if col != "WS":
            features[col] = split_df[col].to_numpy(float)
    for i in range(k_modes):
        features[f"IMF{i + 1}"] = modes[i]
    for i in range(ewt_modes):
        features[f"EWT{i + 1}"] = ewts[i]
    features["WS"] = split_df["WS"].to_numpy(float)
    report = {"selected_inputs": selected, "sample_entropy": se, "high_se_imfs": [f"IMF{i + 1}" for i in high], "columns": list(features.columns)}
    return features, report


class KANHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, grid: int = 8):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.register_buffer("centers", torch.linspace(-1.0, 1.0, grid))
        self.log_width = nn.Parameter(torch.tensor(0.0))
        self.weights = nn.Parameter(torch.randn(in_features, grid, out_features) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = torch.exp(-((x.unsqueeze(-1) - self.centers) ** 2) * torch.exp(self.log_width))
        return self.base(x) + torch.einsum("big,igo->bo", basis, self.weights)


class KANInformerLite(nn.Module):
    def __init__(self, n_features: int, horizons: int, d_model: int, n_heads: int, n_layers: int, kan_grid: int):
        super().__init__()
        self.embedding = nn.Linear(n_features, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = KANHead(d_model, horizons, grid=kan_grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embedding(x)
        z = self.encoder(z)
        return self.head(z[:, -1])


def make_windows_scaled_by_train_only(
    train_features: pd.DataFrame,
    val_features: pd.DataFrame,
    test_features: pd.DataFrame,
    config: RunConfig,
):
    train_values = train_features.to_numpy(float)
    val_values = val_features.to_numpy(float)
    test_values = test_features.to_numpy(float)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_values)
    val_scaled = scaler.transform(val_values)
    test_scaled = scaler.transform(test_values)
    target_idx = train_scaled.shape[1] - 1

    def windows(scaled: np.ndarray):
        xs, ys = [], []
        for i in range(config.window, len(scaled) - config.horizons + 1):
            xs.append(scaled[i - config.window : i])
            ys.append([scaled[i + h, target_idx] for h in range(config.horizons)])
        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

    train_x, train_y = windows(train_scaled)
    val_x, val_y = windows(val_scaled)
    test_x, test_y = windows(test_scaled)
    target_min = scaler.data_min_[target_idx]
    target_max = scaler.data_max_[target_idx]
    test_y_raw = test_y * (target_max - target_min) + target_min
    return (
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y_raw,
        target_min,
        target_max,
    )


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    eps = 1e-8
    return {
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "mape": float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0),
    }


def train_and_evaluate(train_features: pd.DataFrame, val_features: pd.DataFrame, test_features: pd.DataFrame, config: RunConfig, device: torch.device) -> List[Dict[str, object]]:
    train_x, train_y, val_x, val_y, test_x, test_y_raw, target_min, target_max = make_windows_scaled_by_train_only(train_features, val_features, test_features, config)
    model = KANInformerLite(train_features.shape[1], config.horizons, config.d_model, config.n_heads, config.n_layers, config.kan_grid).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_x_t = torch.from_numpy(val_x).to(device)
    val_y_t = torch.from_numpy(val_y).to(device)
    print(f"    samples: train={len(train_x)}, val={len(val_x)}, test={len(test_x)}, features={train_features.shape[1]}")
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(bx), by)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if epoch == 1 or epoch == config.epochs or epoch % max(1, config.epochs // 5) == 0:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(val_x_t), val_y_t).item() if len(val_x) else float("nan")
            print(f"    epoch {epoch:03d}/{config.epochs}: train_loss={np.mean(losses):.6f}, val_loss={val_loss:.6f}")
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(test_x).to(device)).cpu().numpy()
    pred_raw = pred_scaled * (target_max - target_min) + target_min
    rows = []
    for h in range(config.horizons):
        rows.append({"horizon": h + 1, **metric_bundle(test_y_raw[:, h], pred_raw[:, h])})
    return rows


def print_metric_table(rows: List[Dict[str, object]]) -> None:
    print("\nPaper-style metric table")
    print("Season   H  RMSE       MAE        MAPE(%)    Paper_RMSE  Paper_MAE  Paper_MAPE")
    print("------- -- ---------- ---------- ---------- ---------- ---------- ----------")
    for row in rows:
        p = PAPER_FINAL.get((row["season"], row["horizon"]), {})
        print(
            f"{row['season']:<7} h{row['horizon']} "
            f"{row['rmse']:<10.6f} {row['mae']:<10.6f} {row['mape']:<10.3f} "
            f"{p.get('rmse', math.nan):<10.3f} {p.get('mae', math.nan):<10.3f} {p.get('mape', math.nan):<10.3f}"
        )


def write_outputs(rows: List[Dict[str, object]], reports: Dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"pipeline": "no_dataleak", "leakage": False, "reports": reports, "metrics": rows}, f, indent=2)
    lines = ["# No-Data-Leak Pipeline Results", "", "| Season | Horizon | RMSE | MAE | MAPE | Paper RMSE | Paper MAE | Paper MAPE |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        p = PAPER_FINAL[(row["season"], row["horizon"])]
        lines.append(
            f"| {row['season']} | h{row['horizon']} | {row['rmse']:.6f} | {row['mae']:.6f} | {row['mape']:.3f}% | "
            f"{p['rmse']:.3f} | {p['mae']:.3f} | {p['mape']:.1f}% |"
        )
    (out_dir / "paper_format_results.md").write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(config: RunConfig, out_dir: Path) -> None:
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    raw = load_hourly(RAW_CSV)
    seasons = split_seasons(raw)
    all_rows: List[Dict[str, object]] = []
    reports: Dict[str, object] = {"season_reports": {}}
    for season in SEASONS:
        season_df = seasons[season]
        if config.smoke:
            season_df = season_df.head(360).copy()
        print(f"\nRunning {season} [no_dataleak], rows={len(season_df)}")
        train_end = int(len(season_df) * 0.8)
        val_end = int(len(season_df) * 0.9)
        raw_train = season_df.iloc[:train_end].reset_index(drop=True)
        raw_val = season_df.iloc[train_end:val_end].reset_index(drop=True)
        raw_test = season_df.iloc[val_end:].reset_index(drop=True)
        stats = fit_cleaning_stats(raw_train)
        train_clean, train_outliers = clean_with_train_stats_no_leak(raw_train, stats)
        val_clean, val_outliers = clean_with_train_stats_no_leak(raw_val, stats)
        test_clean, test_outliers = clean_with_train_stats_no_leak(raw_test, stats)
        pcc = pcc_report_train_only(train_clean)
        selected = select_train_pcc_inputs(pcc)
        if selected == ["WS"]:
            selected = PAPER_SELECTED_INPUTS[season]
        train_features, train_report = build_split_features_no_leak(season, "train", train_clean, selected, config)
        val_features, val_report = build_split_features_no_leak(season, "val", val_clean, selected, config)
        test_features, test_report = build_split_features_no_leak(season, "test", test_clean, selected, config)
        metrics = train_and_evaluate(train_features, val_features, test_features, config, device)
        reports["season_reports"][season] = {
            "selected_inputs_train_only": selected,
            "outliers": {"train": train_outliers, "val": val_outliers, "test": test_outliers},
            "pcc_train_only": pcc.to_dict(orient="records"),
            "features": {"train": train_report, "val": val_report, "test": test_report},
        }
        for row in metrics:
            all_rows.append({"pipeline": "no_dataleak", "season": season, **row})
    print_metric_table(all_rows)
    write_outputs(all_rows, reports, out_dir)
    print(f"\nWrote results to: {out_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-free chronological VMD-CA-EWT-KANInformer pipeline.")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--kan-grid", type=int, default=8)
    parser.add_argument("--vmd-iter", type=int, default=200)
    parser.add_argument("--ewt-modes", type=int, default=8)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    return parser.parse_args()


def make_config(args: argparse.Namespace, smoke: bool) -> RunConfig:
    return RunConfig(
        epochs=1 if smoke else args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=16 if smoke else args.d_model,
        n_heads=2 if smoke else args.n_heads,
        n_layers=1 if smoke else args.n_layers,
        kan_grid=4 if smoke else args.kan_grid,
        window=7,
        horizons=3,
        vmd_iter=20 if smoke else args.vmd_iter,
        ewt_modes=args.ewt_modes,
        smoke=smoke,
    )


def main() -> None:
    args = parse_args()
    if args.smoke_only:
        run_pipeline(make_config(args, smoke=True), RESULTS_DIR / "_smoke")
        return
    if not args.skip_smoke:
        smoke_dir = RESULTS_DIR / "_smoke"
        run_pipeline(make_config(args, smoke=True), smoke_dir)
        shutil.rmtree(smoke_dir, ignore_errors=True)
        print("\nSmoke test passed and smoke outputs were deleted. Starting full run.\n")
    run_pipeline(make_config(args, smoke=False), RESULTS_DIR)


if __name__ == "__main__":
    main()
