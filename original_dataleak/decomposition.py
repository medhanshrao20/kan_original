"""Established VMD and empirical Meyer wavelets; no hard FFT band approximation."""
import numpy as np
from ewtpy import EWT1D
from scipy.spatial import cKDTree
from vmdpy import VMD


def sample_entropy(x, m=2, r_ratio=0.2):
    x = np.asarray(x, dtype=float)
    if len(x) <= m + 1 or np.std(x) == 0:
        return 0.0
    # Use the same eligible template starts for m and m+1; exclude self matches.
    templates = np.lib.stride_tricks.sliding_window_view(x, m + 1)
    radius = r_ratio * np.std(x)
    count = len(templates)
    b = (cKDTree(templates[:, :m]).count_neighbors(cKDTree(templates[:, :m]), radius, p=np.inf) - count) / 2
    a = (cKDTree(templates).count_neighbors(cKDTree(templates), radius, p=np.inf) - count) / 2
    if b == 0 or a == 0:
        return float("inf")
    return float(-np.log(a / b))


def decompose(ws, k, config, fixed_high=None):
    x = np.asarray(ws, dtype=float)
    if len(x) < max(16, k * 2):
        raise ValueError("Too few observations for VMD/EWT. Increase --decomposition-context.")
    # vmdpy drops odd final samples. Pad first, then crop to preserve hourly alignment.
    padded = np.pad(x, (0, len(x) % 2), mode="edge")
    modes, _, omega = VMD(padded, config.vmd_alpha, config.vmd_tau, k, 0, 1, config.vmd_tol)
    order = np.argsort(omega[-1])
    modes = modes[order, :len(x)]
    entropy = np.array([sample_entropy(mode, config.se_m, config.se_r) for mode in modes])
    high = np.flatnonzero(entropy > config.se_threshold).tolist() if fixed_high is None else list(fixed_high)
    low = [i for i in range(k) if i not in high]
    labels = [f"IMF{i + 1}" for i in low]
    boundaries = []
    if high:
        aggregate = modes[high].sum(axis=0)
        _, filters, boundaries = EWT1D(aggregate, N=config.ewt_modes, completion=1,
                                                  detect="locmax", reg="average")
        # Eq. (15)-(16) requires synthesis as well as analysis: F * |filter|^2.
        left = int(np.ceil(len(aggregate) / 2))
        mirrored = np.concatenate([aggregate[:left - 1][::-1], aggregate, aggregate[-left - 1:-1][::-1]])
        spectrum = np.fft.fft(mirrored)
        energy = np.sum(np.abs(filters) ** 2, axis=1, keepdims=True)
        if np.any(energy <= 0):
            raise ValueError("EWT filter bank has uncovered frequencies.")
        # Canonical dual synthesis corrects the package's non-unit finite-grid
        # frame energy at the Nyquist transition; every band still uses Meyer filters.
        reconstructed = np.fft.ifft(spectrum[:, None] * np.abs(filters) ** 2 / energy, axis=0).real
        ewt = reconstructed[left - 1:left - 1 + len(x)].T
        if ewt.shape[0] != config.ewt_modes:
            raise ValueError(f"EWT returned {ewt.shape[0]} modes, expected {config.ewt_modes}")
        parts = np.vstack([modes[low], ewt])
        labels += [f"EWT{i + 1}" for i in range(len(ewt))]
        ewt_error = float(np.max(np.abs(ewt.sum(axis=0) - aggregate)))
    else:
        parts = modes
        labels = [f"IMF{i + 1}" for i in range(k)]
        ewt_error = None
    report = {
        "k": k, "high_indices": high, "high_imfs": [f"IMF{i + 1}" for i in high],
        "entropy": [float(v) if np.isfinite(v) else None for v in entropy],
        "entropy_nonfinite_indices": np.flatnonzero(~np.isfinite(entropy)).tolist(),
        "vmd_relative_absolute_residual": float(np.mean(np.abs(x - modes.sum(axis=0)) / np.maximum(np.abs(x), 1e-12))),
        "ewt_boundaries_radians": np.asarray(boundaries).tolist(),
        "ewt_synthesis_max_error": ewt_error,
        "labels": labels,
    }
    return parts.T, report
