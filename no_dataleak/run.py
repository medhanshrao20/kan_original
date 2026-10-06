from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PIPELINE_NAME = "no_dataleak"
DECOMPOSE_BEFORE_SPLIT = False

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
    e_layers: int
    d_ff: int
    kan_grid: int
    window: int
    horizons: int
    vmd_iter: int
    ewt_modes: int
    se_threshold: float
    smoke: bool


def seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


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


def clean_like_paper(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    print("Paper preprocessing: 3-sigma outlier removal + cubic spline interpolation.")
    cleaned = df.copy()
    counts: Dict[str, int] = {}
    for col in VARIABLES:
        x = cleaned[col].astype(float)
        std = x.std(skipna=True)
        if not np.isfinite(std) or std == 0:
            counts[col] = 0
            continue
        mask = (x - x.mean(skipna=True)).abs() > 3.0 * std
        counts[col] = int(mask.sum())
        cleaned.loc[mask, col] = np.nan

    cleaned = cleaned.set_index("timestamp")
    for col in VARIABLES:
        try:
            cleaned[col] = cleaned[col].interpolate(method="spline", order=3).ffill().bfill()
        except Exception:
            cleaned[col] = cleaned[col].interpolate(method="linear").ffill().bfill()
    return cleaned.reset_index(), counts


def split_seasons(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    ts = df["timestamp"]
    masks = {
        "winter": (ts >= "2020-12-01") & (ts < "2021-03-01"),
        "spring": (ts >= "2021-03-01") & (ts < "2021-06-01"),
        "summer": (ts >= "2021-06-01") & (ts < "2021-09-01"),
        "autumn": (ts >= "2021-09-01") & (ts < "2021-12-01"),
    }
    return {season: df.loc[mask].reset_index(drop=True) for season, mask in masks.items()}


def chronological_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    return (
        df.iloc[:train_end].reset_index(drop=True),
        df.iloc[train_end:val_end].reset_index(drop=True),
        df.iloc[val_end:].reset_index(drop=True),
    )


def pcc_report(df: pd.DataFrame) -> pd.DataFrame:
    ws = df["WS"].to_numpy(float)
    rows = []
    for col in METEO_VARIABLES:
        x = df[col].to_numpy(float)
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


def vmd(signal: np.ndarray, k_modes: int, alpha: float = 2000.0, tau: float = 0.0, tol: float = 1e-7, max_iter: int = 200) -> np.ndarray:
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
        lambda_hat[n + 1] = lambda_hat[n] + tau * (np.sum(u_hat[n + 1], axis=1) - f_hat_plus)
        n += 1
        diff = u_hat[n, :, :] - u_hat[n - 1, :, :]
        u_diff = abs(np.sum(np.conj(diff) * diff) / len(freqs))
    full_hat = np.zeros((len(freqs), k_modes), dtype=complex)
    full_hat[len(freqs) // 2 :] = u_hat[n, len(freqs) // 2 :]
    full_hat[: len(freqs) // 2] = np.conj(np.flipud(u_hat[n, len(freqs) // 2 :]))
    modes = np.vstack([np.real(np.fft.ifft(np.fft.ifftshift(full_hat[:, mode]))) for mode in range(k_modes)])
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
    magnitude = np.abs(spectrum)
    peaks, _ = find_peaks(magnitude)
    if len(peaks) >= n_modes:
        strongest = sorted(peaks[np.argsort(magnitude[peaks])[-n_modes:]])
        boundaries = [int(round((a + b) / 2)) for a, b in zip(strongest[:-1], strongest[1:])]
    else:
        boundaries = np.linspace(1, len(magnitude) - 1, n_modes + 1, dtype=int)[1:-1].tolist()
    edges = [0] + sorted(set(boundaries)) + [len(magnitude)]
    while len(edges) < n_modes + 1:
        edges = sorted(set(edges + np.linspace(0, len(magnitude), n_modes + 1, dtype=int).tolist()))
    edges = edges[:n_modes] + [len(magnitude)]
    parts = []
    for start, end in zip(edges[:-1], edges[1:]):
        band = np.zeros_like(spectrum)
        band[start:end] = spectrum[start:end]
        parts.append(np.fft.irfft(band, n=len(x)))
    return np.vstack(parts)


def fit_vmd_ca_ewt(
    ws: np.ndarray,
    k_modes: int,
    ewt_modes: int,
    se_threshold: float,
    max_iter: int,
    fixed_high_idx: List[int] | None = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    imfs = vmd(ws, k_modes=k_modes, max_iter=max_iter)
    entropy = {f"IMF{i + 1}": sample_entropy(imfs[i]) for i in range(k_modes)}
    if fixed_high_idx is None:
        high_idx = [i for i in range(k_modes) if entropy[f"IMF{i + 1}"] > se_threshold]
        if not high_idx:
            high_idx = [int(np.argmax([entropy[f"IMF{i + 1}"] for i in range(k_modes)]))]
    else:
        high_idx = fixed_high_idx
    low_idx = [i for i in range(k_modes) if i not in high_idx]
    high_sum = imfs[high_idx].sum(axis=0)
    ewt_parts = ewt_style(high_sum, ewt_modes)
    report = {
        "k_modes": k_modes,
        "sample_entropy": entropy,
        "high_complexity_indices_zero_based": high_idx,
        "high_complexity_imfs": [f"IMF{i + 1}" for i in high_idx],
        "retained_low_complexity_imfs": [f"IMF{i + 1}" for i in low_idx],
        "ewt_modes": ewt_modes,
    }
    return np.vstack([imfs[low_idx], ewt_parts]) if low_idx else ewt_parts, report


def build_feature_frame(
    season_df: pd.DataFrame,
    selected_inputs: Sequence[str],
    decomp_components: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(index=season_df.index)
    for col in selected_inputs:
        if col != "WS":
            frame[col] = season_df[col].to_numpy(float)
    for i in range(decomp_components.shape[0]):
        frame[f"DECOMP{i + 1}"] = decomp_components[i]
    frame["WS"] = season_df["WS"].to_numpy(float)
    return frame


class KANHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, grid: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer("centers", torch.linspace(-1.0, 1.0, grid))
        self.log_width = nn.Parameter(torch.tensor(0.0))
        self.weights = nn.Parameter(torch.randn(in_features, grid, out_features) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = torch.exp(-((x.unsqueeze(-1) - self.centers) ** 2) * torch.exp(self.log_width))
        return self.linear(x) + torch.einsum("big,igo->bo", basis, self.weights)


class PaperKANInformer(nn.Module):
    def __init__(self, n_features: int, horizons: int, config: RunConfig):
        super().__init__()
        self.embedding = nn.Linear(n_features, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.e_layers)
        self.kan_projection = KANHead(config.d_model, horizons, config.kan_grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embedding(x)
        z = self.encoder(z)
        return self.kan_projection(z[:, -1])


def make_windows(scaled_values: np.ndarray, window: int, horizons: int) -> Tuple[np.ndarray, np.ndarray]:
    target_idx = scaled_values.shape[1] - 1
    xs, ys = [], []
    for i in range(window, len(scaled_values) - horizons + 1):
        xs.append(scaled_values[i - window : i])
        ys.append([scaled_values[i + h, target_idx] for h in range(horizons)])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def prepare_scaled_windows(
    train_features: pd.DataFrame,
    val_features: pd.DataFrame,
    test_features: pd.DataFrame,
    config: RunConfig,
):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_features.to_numpy(float))
    val_scaled = scaler.transform(val_features.to_numpy(float))
    test_scaled = scaler.transform(test_features.to_numpy(float))
    train_x, train_y = make_windows(train_scaled, config.window, config.horizons)
    val_x, val_y = make_windows(val_scaled, config.window, config.horizons)
    test_x, test_y = make_windows(test_scaled, config.window, config.horizons)
    target_min = scaler.data_min_[-1]
    target_max = scaler.data_max_[-1]
    test_y_raw = test_y * (target_max - target_min) + target_min
    return train_x, train_y, val_x, val_y, test_x, test_y_raw, target_min, target_max


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    denom = np.maximum(np.abs(y_true), 1e-8)
    return {
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "mape": float(np.mean(np.abs((y_pred - y_true) / denom)) * 100.0),
    }


def train_and_evaluate(
    train_features: pd.DataFrame,
    val_features: pd.DataFrame,
    test_features: pd.DataFrame,
    config: RunConfig,
    device: torch.device,
) -> List[Dict[str, object]]:
    train_x, train_y, val_x, val_y, test_x, test_y_raw, target_min, target_max = prepare_scaled_windows(
        train_features, val_features, test_features, config
    )
    print(f"    windows: train={len(train_x)}, val={len(val_x)}, test={len(test_x)}, features={train_features.shape[1]}")
    model = PaperKANInformer(train_features.shape[1], config.horizons, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_x_t = torch.from_numpy(val_x).to(device)
    val_y_t = torch.from_numpy(val_y).to(device)
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
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
    return [{"horizon": i + 1, **metric_bundle(test_y_raw[:, i], pred_raw[:, i])} for i in range(config.horizons)]


def decompose_before_split(
    season: str,
    season_df: pd.DataFrame,
    config: RunConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    selected = PAPER_SELECTED_INPUTS[season]
    k_modes = min(3, PAPER_VMD_K[season]) if config.smoke else PAPER_VMD_K[season]
    ewt_modes = min(3, config.ewt_modes) if config.smoke else config.ewt_modes
    print(f"  {season}: VMD-CA-EWT on FULL season before split [data-leakage test], K={k_modes}, EWT={ewt_modes}")
    components, decomp_report = fit_vmd_ca_ewt(
        season_df["WS"].to_numpy(float),
        k_modes=k_modes,
        ewt_modes=ewt_modes,
        se_threshold=config.se_threshold,
        max_iter=config.vmd_iter,
    )
    full_features = build_feature_frame(season_df, selected, components)
    train_f, val_f, test_f = chronological_split(full_features)
    return train_f, val_f, test_f, {"selected_inputs": selected, "decomposition": {"full_season": decomp_report}}


def decompose_after_split(
    season: str,
    season_df: pd.DataFrame,
    config: RunConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    selected = PAPER_SELECTED_INPUTS[season]
    train_df, val_df, test_df = chronological_split(season_df)
    k_modes = min(3, PAPER_VMD_K[season]) if config.smoke else PAPER_VMD_K[season]
    ewt_modes = min(3, config.ewt_modes) if config.smoke else config.ewt_modes
    print(f"  {season}: VMD-CA-EWT separately AFTER split [no leakage], K={k_modes}, EWT={ewt_modes}")
    train_components, train_report = fit_vmd_ca_ewt(
        train_df["WS"].to_numpy(float), k_modes, ewt_modes, config.se_threshold, config.vmd_iter
    )
    train_high_idx = train_report["high_complexity_indices_zero_based"]
    val_components, val_report = fit_vmd_ca_ewt(
        val_df["WS"].to_numpy(float), k_modes, ewt_modes, config.se_threshold, config.vmd_iter, fixed_high_idx=train_high_idx
    )
    test_components, test_report = fit_vmd_ca_ewt(
        test_df["WS"].to_numpy(float), k_modes, ewt_modes, config.se_threshold, config.vmd_iter, fixed_high_idx=train_high_idx
    )
    train_f = build_feature_frame(train_df, selected, train_components)
    val_f = build_feature_frame(val_df, selected, val_components)
    test_f = build_feature_frame(test_df, selected, test_components)
    return train_f, val_f, test_f, {
        "selected_inputs": selected,
        "decomposition": {"train": train_report, "val": val_report, "test": test_report},
    }


def print_metric_table(rows: List[Dict[str, object]]) -> None:
    print("\nPaper-format metric table")
    print("Season   H  RMSE       MAE        MAPE(%)    Paper_RMSE  Paper_MAE  Paper_MAPE")
    print("------- -- ---------- ---------- ---------- ---------- ---------- ----------")
    for row in rows:
        paper = PAPER_FINAL[(row["season"], row["horizon"])]
        print(
            f"{row['season']:<7} h{row['horizon']} "
            f"{row['rmse']:<10.6f} {row['mae']:<10.6f} {row['mape']:<10.3f} "
            f"{paper['rmse']:<10.3f} {paper['mae']:<10.3f} {paper['mape']:<10.3f}"
        )


def write_outputs(rows: List[Dict[str, object]], reports: Dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
    payload = {
        "pipeline": PIPELINE_NAME,
        "decompose_before_split": DECOMPOSE_BEFORE_SPLIT,
        "only_intended_difference": "VMD-CA-EWT before split vs after split",
        "reports": reports,
        "metrics": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# {PIPELINE_NAME} Results",
        "",
        "| Season | Horizon | RMSE | MAE | MAPE | Paper RMSE | Paper MAE | Paper MAPE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        paper = PAPER_FINAL[(row["season"], row["horizon"])]
        lines.append(
            f"| {row['season']} | h{row['horizon']} | {row['rmse']:.6f} | {row['mae']:.6f} | {row['mape']:.3f}% | "
            f"{paper['rmse']:.3f} | {paper['mae']:.3f} | {paper['mape']:.1f}% |"
        )
    (out_dir / "paper_format_results.md").write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(config: RunConfig, out_dir: Path) -> None:
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Pipeline: {PIPELINE_NAME}")
    print(f"Decomposition before split: {DECOMPOSE_BEFORE_SPLIT}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    raw = load_hourly(RAW_CSV)
    cleaned, outliers = clean_like_paper(raw)
    seasons = split_seasons(cleaned)
    rows: List[Dict[str, object]] = []
    reports: Dict[str, object] = {"outlier_counts_3sigma": outliers, "season_reports": {}}
    for season in SEASONS:
        season_df = seasons[season]
        if config.smoke:
            season_df = season_df.head(360).copy()
        print(f"\nRunning {season}, rows={len(season_df)}")
        pcc = pcc_report(season_df)
        if DECOMPOSE_BEFORE_SPLIT:
            train_f, val_f, test_f, report = decompose_before_split(season, season_df, config)
        else:
            train_f, val_f, test_f, report = decompose_after_split(season, season_df, config)
        metrics = train_and_evaluate(train_f, val_f, test_f, config, device)
        report["pcc_full_season_report"] = pcc.to_dict(orient="records")
        report["feature_columns"] = list(train_f.columns)
        reports["season_reports"][season] = report
        for row in metrics:
            rows.append({"pipeline": PIPELINE_NAME, "season": season, **row})
    print_metric_table(rows)
    write_outputs(rows, reports, out_dir)
    print(f"\nWrote results to: {out_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-replica KANInformer pipeline with VMD-CA-EWT after split.")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--e-layers", type=int, default=1)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--kan-grid", type=int, default=5)
    parser.add_argument("--vmd-iter", type=int, default=200)
    parser.add_argument("--ewt-modes", type=int, default=8)
    parser.add_argument("--se-threshold", type=float, default=0.4)
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
        e_layers=1 if smoke else args.e_layers,
        d_ff=32 if smoke else args.d_ff,
        kan_grid=3 if smoke else args.kan_grid,
        window=7,
        horizons=3,
        vmd_iter=20 if smoke else args.vmd_iter,
        ewt_modes=args.ewt_modes,
        se_threshold=args.se_threshold,
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
