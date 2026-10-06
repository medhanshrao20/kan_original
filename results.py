from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "original_dataleak" / "results" / "metrics.csv"
NO_LEAK = ROOT / "no_dataleak" / "results" / "metrics.csv"
OUT_DIR = ROOT / "combined_results"

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


def load_metrics(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} results: {path}. Run that folder's run.py first.")
    frame = pd.read_csv(path)
    required = {"season", "horizon", "rmse", "mae", "mape"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    original = load_metrics(ORIGINAL, "original_dataleak").copy()
    no_leak = load_metrics(NO_LEAK, "no_dataleak").copy()

    original = original.rename(columns={"rmse": "original_dataleak_rmse", "mae": "original_dataleak_mae", "mape": "original_dataleak_mape"})
    no_leak = no_leak.rename(columns={"rmse": "no_dataleak_rmse", "mae": "no_dataleak_mae", "mape": "no_dataleak_mape"})

    cols_original = ["season", "horizon", "original_dataleak_rmse", "original_dataleak_mae", "original_dataleak_mape"]
    cols_no_leak = ["season", "horizon", "no_dataleak_rmse", "no_dataleak_mae", "no_dataleak_mape"]
    merged = original[cols_original].merge(no_leak[cols_no_leak], on=["season", "horizon"], how="outer")

    for metric in ["rmse", "mae", "mape"]:
        merged[f"paper_{metric}"] = [PAPER_FINAL[(s, int(h))][metric] for s, h in zip(merged["season"], merged["horizon"])]
        merged[f"original_gap_{metric}"] = merged[f"original_dataleak_{metric}"] - merged[f"paper_{metric}"]
        merged[f"no_leak_gap_{metric}"] = merged[f"no_dataleak_{metric}"] - merged[f"paper_{metric}"]

    merged = merged.sort_values(["season", "horizon"]).reset_index(drop=True)
    merged.to_csv(OUT_DIR / "dataleak_vs_no_dataleak_comparison.csv", index=False)

    summary = {
        "original_dataleak_metrics": str(ORIGINAL),
        "no_dataleak_metrics": str(NO_LEAK),
        "comparison_csv": str(OUT_DIR / "dataleak_vs_no_dataleak_comparison.csv"),
        "interpretation": "Lower RMSE/MAE/MAPE is better. If original_dataleak is much closer to paper values than no_dataleak, that supports the user's leakage hypothesis.",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Data-Leak vs No-Data-Leak Comparison",
        "",
        "| Season | Horizon | Paper RMSE | Original Leak RMSE | No Leak RMSE | Paper MAE | Original Leak MAE | No Leak MAE | Paper MAPE | Original Leak MAPE | No Leak MAPE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in merged.iterrows():
        lines.append(
            f"| {row['season']} | h{int(row['horizon'])} | "
            f"{row['paper_rmse']:.6f} | {row['original_dataleak_rmse']:.6f} | {row['no_dataleak_rmse']:.6f} | "
            f"{row['paper_mae']:.6f} | {row['original_dataleak_mae']:.6f} | {row['no_dataleak_mae']:.6f} | "
            f"{row['paper_mape']:.3f}% | {row['original_dataleak_mape']:.3f}% | {row['no_dataleak_mape']:.3f}% |"
        )
    (OUT_DIR / "comparison.md").write_text("\n".join(lines), encoding="utf-8")

    print("\nCombined comparison")
    print(merged[[
        "season",
        "horizon",
        "paper_rmse",
        "original_dataleak_rmse",
        "no_dataleak_rmse",
        "paper_mape",
        "original_dataleak_mape",
        "no_dataleak_mape",
    ]].to_string(index=False))
    print(f"\nWrote combined outputs to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
