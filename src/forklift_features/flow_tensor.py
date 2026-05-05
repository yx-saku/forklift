from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_MOTION_FEATURE_NAMES = ("flow_x", "flow_y")
SUPPORTED_MOTION_FEATURE_NAMES = (
    "flow_x",
    "flow_y",
    "t_flow_x",
    "t_flow_y",
    "flow_x_broad_vib_score",
    "flow_y_broad_vib_score",
    "t_flow_x_broad_vib_score",
    "t_flow_y_broad_vib_score",
)
BROAD_VIBRATION_FEATURE_TO_BASE = {
    "flow_x_broad_vib_score": "flow_x",
    "flow_y_broad_vib_score": "flow_y",
    "t_flow_x_broad_vib_score": "t_flow_x",
    "t_flow_y_broad_vib_score": "t_flow_y",
}
BASE_SEQUENCE_FEATURE_CHANNELS = {
    "flow_x": 0,
    "flow_y": 1,
    "t_flow_x": 2,
    "t_flow_y": 3,
}
DEFAULT_BROAD_VIBRATION_SCORE_CONFIG = {
    "high_ratio_fraction": 0.5,
    "lower_percentile": 0.0,
    "upper_percentile": 95.0,
    "min_visible": 1e-6,
    "broad_vib_score_weights": {"A": 0.25, "B": 0.75, "C": 0.0},
    "broad_vib_low_intensity_percentile": 20.0,
}


def _numeric_column(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default).astype(float)


def normalize_motion_feature_names(feature_names: Iterable[str] | None = None) -> list[str]:
    selected = list(DEFAULT_MOTION_FEATURE_NAMES if feature_names is None else feature_names)
    normalized = []
    for name in selected:
        feature_name = str(name).strip()
        if not feature_name:
            continue
        if feature_name not in SUPPORTED_MOTION_FEATURE_NAMES:
            raise ValueError(f"Unsupported motion feature: {feature_name}. supported={list(SUPPORTED_MOTION_FEATURE_NAMES)}")
        if feature_name not in normalized:
            normalized.append(feature_name)
    if not normalized:
        raise ValueError("At least one motion feature must be selected")
    return normalized


def _broad_vibration_score_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(DEFAULT_BROAD_VIBRATION_SCORE_CONFIG)
    if config:
        merged.update(dict(config))
    weights = dict(DEFAULT_BROAD_VIBRATION_SCORE_CONFIG["broad_vib_score_weights"])
    weights.update(dict(merged.get("broad_vib_score_weights", {}) or {}))
    merged["broad_vib_score_weights"] = weights
    return merged


def _robust_positive_zscore(values: np.ndarray | pd.Series) -> np.ndarray:
    values_arr = np.asarray(values, dtype=float)
    if values_arr.size == 0:
        return np.zeros((0,), dtype=float)
    finite = values_arr[np.isfinite(values_arr)]
    if finite.size == 0:
        return np.zeros_like(values_arr, dtype=float)
    median = float(np.median(finite))
    scale = max(1.4826 * float(np.median(np.abs(finite - median))), 1e-6)
    return np.maximum((np.nan_to_num(values_arr, nan=median) - median) / scale, 0.0)


def _percentile_normalize_0_1(
    values: np.ndarray | pd.Series,
    *,
    lower_percentile: float = 0.0,
    upper_percentile: float = 95.0,
    min_visible: float = 1e-6,
) -> np.ndarray:
    values_arr = np.asarray(values, dtype=float)
    out = np.zeros_like(values_arr, dtype=float)
    finite_mask = np.isfinite(values_arr)
    if not finite_mask.any():
        return out
    fit_values = values_arr[finite_mask]
    fit_values = fit_values[fit_values > float(min_visible)]
    if fit_values.size == 0:
        return out
    lo_q, hi_q = sorted([float(np.clip(lower_percentile, 0.0, 100.0)), float(np.clip(upper_percentile, 0.0, 100.0))])
    lo, hi = float(np.percentile(fit_values, lo_q)), float(np.percentile(fit_values, hi_q))
    if hi <= lo:
        out[finite_mask] = np.where(values_arr[finite_mask] > min_visible, 1.0, 0.0)
    else:
        out[finite_mask] = np.clip((values_arr[finite_mask] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return np.where(values_arr > min_visible, out, 0.0).astype(float)


def _significant_turn_ratio(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return 0.0
    diffs = np.diff(values)
    finite_diffs = diffs[np.isfinite(diffs)]
    if finite_diffs.size < 2:
        return 0.0
    threshold = max(float(np.percentile(np.abs(finite_diffs), 50)) * 0.25, 1e-9)
    significant = np.abs(diffs) >= threshold
    turn_mask = significant[1:] & significant[:-1] & (diffs[1:] * diffs[:-1] < 0.0)
    denominator = max(1, int(np.count_nonzero(significant)) - 1)
    return float(np.count_nonzero(turn_mask) / denominator)


def _significant_direction_balance(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    diffs = np.diff(values)
    finite_diffs = diffs[np.isfinite(diffs)]
    if finite_diffs.size == 0:
        return 0.0
    threshold = max(float(np.percentile(np.abs(finite_diffs), 50)) * 0.25, 1e-9)
    positive_sum = float(np.sum(finite_diffs[finite_diffs >= threshold]))
    negative_sum = float(np.sum(np.abs(finite_diffs[finite_diffs <= -threshold])))
    denominator = max(positive_sum, negative_sum)
    return float(min(positive_sum, negative_sum) / denominator) if denominator > 1e-12 else 0.0


def _build_oscillation_window_features(segment: np.ndarray, high_ratio_fraction: float) -> dict[str, float]:
    values = np.nan_to_num(np.asarray(segment, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if values.size == 0:
        values = np.zeros(1, dtype=float)
    center = float(np.median(values))
    residual_abs = np.abs(values - center)
    signed_diffs = np.diff(values) if values.size >= 2 else np.zeros(0, dtype=float)
    diffs = np.abs(signed_diffs)
    diff_max = float(np.max(diffs)) if diffs.size else 0.0
    diff_high_ratio = float(np.mean(diffs >= diff_max * high_ratio_fraction)) if diff_max > 1e-6 and diffs.size else 0.0
    range_p90_p10 = float(np.percentile(values, 90) - np.percentile(values, 10)) if values.size else 0.0
    residual_mean = float(np.mean(residual_abs)) if residual_abs.size else 0.0
    diff_sum = float(np.sum(diffs)) if diffs.size else 0.0
    diff_p95 = float(np.percentile(diffs, 95)) if diffs.size else 0.0
    direction_balance = _significant_direction_balance(values)
    turn_ratio = _significant_turn_ratio(values)
    return {
        "change_sum": float(np.sum(residual_abs) + diff_sum),
        "osc_range_p90_p10": range_p90_p10,
        "osc_residual_mean": residual_mean,
        "osc_diff_sum": diff_sum,
        "osc_balanced_diff_sum": diff_sum * direction_balance,
        "osc_turn_ratio": turn_ratio,
        "change_p95": max(float(np.percentile(residual_abs, 95)) if residual_abs.size else 0.0, diff_p95),
        "change_max": max(float(np.max(residual_abs)) if residual_abs.size else 0.0, diff_max),
        "change_high_ratio": diff_high_ratio,
    }


def _add_oscillation_vibration_score_columns(window_df: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    work = window_df.copy()
    weights = {
        "osc_range_p90_p10": 0.20,
        "osc_residual_mean": 0.20,
        "osc_diff_sum": 0.15,
        "osc_balanced_diff_sum": 0.30,
        "osc_turn_ratio": 0.15,
    }
    for name in weights:
        if name not in work.columns:
            work[name] = 0.0
        work[f"{name}_z"] = _robust_positive_zscore(work[name])
    work["vibration_score_raw"] = sum(weight * work[f"{name}_z"] for name, weight in weights.items()).astype(float)
    work["vibration_score"] = _percentile_normalize_0_1(
        work["vibration_score_raw"],
        lower_percentile=float(config.get("lower_percentile", 0.0)),
        upper_percentile=float(config.get("upper_percentile", 95.0)),
        min_visible=float(config.get("min_visible", 1e-6)),
    )
    return work


def _build_broad_vibration_lookup(
    sequence: np.ndarray,
    starts: list[int],
    time_steps: int,
    feature_names: list[str],
    config: Mapping[str, Any] | None,
) -> dict[tuple[str, int], float]:
    requested = [name for name in feature_names if name in BROAD_VIBRATION_FEATURE_TO_BASE]
    if not requested or sequence.shape[0] == 0:
        return {}
    cfg = _broad_vibration_score_config(config)
    high_ratio_fraction = float(cfg.get("high_ratio_fraction", 0.5))
    weights = cfg.get("broad_vib_score_weights", {}) or {}
    weight_a = float(weights.get("A", weights.get("a", 0.25)))
    weight_b = float(weights.get("B", weights.get("b", 0.75)))
    weight_c = float(weights.get("C", weights.get("c", 0.0)))
    low_intensity_percentile = float(np.clip(cfg.get("broad_vib_low_intensity_percentile", 20.0), 0.0, 100.0))
    lookup: dict[tuple[str, int], float] = {}
    for feature_name in requested:
        base_feature = BROAD_VIBRATION_FEATURE_TO_BASE[feature_name]
        channel_index = BASE_SEQUENCE_FEATURE_CHANNELS[base_feature]
        rows: list[dict[str, float]] = []
        for gy in range(sequence.shape[1]):
            for gx in range(sequence.shape[2]):
                cell_rows: list[dict[str, float]] = []
                for start in starts:
                    end = min(start + time_steps, sequence.shape[0])
                    segment = sequence[start:end, gy, gx, channel_index]
                    cell_rows.append({
                        "start": int(start),
                        "grid_row": int(gy),
                        "grid_col": int(gx),
                        **_build_oscillation_window_features(segment, high_ratio_fraction),
                    })
                if cell_rows:
                    rows.extend(_add_oscillation_vibration_score_columns(pd.DataFrame(cell_rows), cfg).to_dict("records"))
        vibration_df = pd.DataFrame(rows)
        if vibration_df.empty:
            continue
        for start, group in vibration_df.groupby("start", sort=False):
            scores = pd.to_numeric(group["vibration_score"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
            change_sums = pd.to_numeric(group["change_sum"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
            intensities = scores * change_sums
            mean_intensity = float(np.mean(intensities)) if intensities.size else 0.0
            low_intensity = float(np.percentile(intensities, low_intensity_percentile)) if intensities.size else 0.0
            max_intensity = float(np.max(intensities)) if intensities.size else 0.0
            value_sum = float(np.sum(intensities)) if intensities.size else 0.0
            value_square_sum = float(np.sum(intensities * intensities)) if intensities.size else 0.0
            effective_cells = (value_sum * value_sum / value_square_sum) if value_square_sum > 1e-12 else 0.0
            grid_count = int(len(group))
            spread_score = float(np.clip(effective_cells / grid_count, 0.0, 1.0)) if grid_count else 0.0
            lookup[(feature_name, int(start))] = float(
                weight_a * mean_intensity * spread_score
                + weight_b * low_intensity
                + weight_c * max_intensity
            )
    return lookup


def _assemble_motion_feature_window(
    raw_window: np.ndarray,
    *,
    start: int,
    feature_names: list[str],
    broad_vibration_lookup: Mapping[tuple[str, int], float],
) -> np.ndarray:
    parts = []
    for feature_name in feature_names:
        if feature_name == "flow_x":
            parts.append(raw_window[..., 0:1])
        elif feature_name == "flow_y":
            parts.append(raw_window[..., 1:2])
        elif feature_name == "t_flow_x":
            parts.append(raw_window[..., 2:3])
        elif feature_name == "t_flow_y":
            parts.append(raw_window[..., 3:4])
        elif feature_name in BROAD_VIBRATION_FEATURE_TO_BASE:
            value = float(broad_vibration_lookup.get((feature_name, int(start)), 0.0))
            parts.append(np.full((*raw_window.shape[:-1], 1), value, dtype=np.float32))
        else:
            raise ValueError(f"Unsupported motion feature: {feature_name}")
    return np.concatenate(parts, axis=-1).astype(np.float32, copy=False)


def raw_flow_to_camera_grid_sequence(
    raw_flow_df: pd.DataFrame,
    camera: str,
    *,
    grid: tuple[int, int] = (3, 3),
) -> tuple[np.ndarray, np.ndarray]:
    """Convert wide raw-flow rows into base ``flow_x/flow_y/t_flow_x/t_flow_y`` sequence."""
    if raw_flow_df is None or raw_flow_df.empty or "time" not in raw_flow_df.columns:
        return np.zeros((0,), dtype=float), np.zeros((0, grid[0], grid[1], len(BASE_SEQUENCE_FEATURE_CHANNELS)), dtype=np.float32)

    prefix = str(camera).lower().strip()
    work = raw_flow_df.copy()
    work["time"] = _numeric_column(work, "time", 0.0)
    work = work.sort_values("time")
    if work["time"].duplicated().any():
        numeric_cols = [col for col in work.select_dtypes(include=[np.number]).columns.tolist() if col != "time"]
        work = work.groupby("time", as_index=False)[numeric_cols].mean()

    times = work["time"].to_numpy(dtype=float)
    sequence = np.zeros((len(work), int(grid[0]), int(grid[1]), len(BASE_SEQUENCE_FEATURE_CHANNELS)), dtype=np.float32)
    for gy in range(int(grid[0])):
        for gx in range(int(grid[1])):
            x_col = f"{prefix}_flow_cell_{gy}_{gx}_x_mean"
            y_col = f"{prefix}_flow_cell_{gy}_{gx}_y_mean"
            reliable_col = f"{prefix}_flow_cell_{gy}_{gx}_flow_reliable_ratio"
            valid_col = f"{prefix}_flow_cell_{gy}_{gx}_valid_ratio"
            x = _numeric_column(work, x_col, 0.0).to_numpy(dtype=np.float32)
            y = _numeric_column(work, y_col, 0.0).to_numpy(dtype=np.float32)
            if reliable_col in work.columns:
                reliable = _numeric_column(work, reliable_col, 0.0).clip(0.0, 1.0).to_numpy(dtype=np.float32)
            elif valid_col in work.columns:
                reliable = _numeric_column(work, valid_col, 0.0).clip(0.0, 1.0).to_numpy(dtype=np.float32)
            else:
                reliable = np.ones(len(work), dtype=np.float32)
            sequence[:, gy, gx, 0] = x
            sequence[:, gy, gx, 1] = y
            sequence[:, gy, gx, 2] = x * reliable
            sequence[:, gy, gx, 3] = y * reliable
    return times, sequence


def _window_starts(length: int, time_steps: int, hop_steps: int) -> list[int]:
    if length <= 0:
        return []
    if length <= time_steps:
        return [0]
    starts = list(range(0, length - time_steps + 1, max(1, hop_steps)))
    final_start = length - time_steps
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def build_flow_tensor_windows(
    raw_flow_df: pd.DataFrame,
    *,
    cameras: Iterable[str] = ("front", "rear"),
    flow_sample_sec: float,
    window_sec: float,
    hop_sec: float,
    grid: tuple[int, int] = (3, 3),
    feature_names: Iterable[str] | None = None,
    broad_vib_score_config: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build camera-wise flow tensors with shape ``(N, T, 3, 3, C)``."""
    motion_feature_names = normalize_motion_feature_names(feature_names)
    time_steps = max(1, int(round(float(window_sec) / max(float(flow_sample_sec), 1e-6))))
    hop_steps = max(1, int(round(float(hop_sec) / max(float(flow_sample_sec), 1e-6))))
    sample_id = str(raw_flow_df["video_id"].dropna().iat[0]) if raw_flow_df is not None and len(raw_flow_df) and "video_id" in raw_flow_df.columns and raw_flow_df["video_id"].dropna().size else "unknown"
    target_category = str(raw_flow_df["target_category"].dropna().iat[0]) if raw_flow_df is not None and "target_category" in raw_flow_df.columns and raw_flow_df["target_category"].dropna().size else "unknown"
    target_environment = str(raw_flow_df["target_environment"].dropna().iat[0]) if raw_flow_df is not None and "target_environment" in raw_flow_df.columns and raw_flow_df["target_environment"].dropna().size else "unknown"
    target_label = str(raw_flow_df["target_label"].dropna().iat[0]) if raw_flow_df is not None and "target_label" in raw_flow_df.columns and raw_flow_df["target_label"].dropna().size else sample_id

    tensors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for camera in cameras:
        times, sequence = raw_flow_to_camera_grid_sequence(raw_flow_df, str(camera), grid=grid)
        if sequence.shape[0] == 0:
            continue
        starts = _window_starts(sequence.shape[0], time_steps, hop_steps)
        broad_vibration_lookup = _build_broad_vibration_lookup(
            sequence,
            starts,
            time_steps,
            motion_feature_names,
            broad_vib_score_config,
        )
        for start in starts:
            end = min(start + time_steps, sequence.shape[0])
            window = sequence[start:end]
            if window.shape[0] < time_steps:
                pad = np.zeros((time_steps - window.shape[0], int(grid[0]), int(grid[1]), sequence.shape[-1]), dtype=np.float32)
                window = np.concatenate([window, pad], axis=0)
            window = _assemble_motion_feature_window(
                window.astype(np.float32, copy=False),
                start=start,
                feature_names=motion_feature_names,
                broad_vibration_lookup=broad_vibration_lookup,
            )
            start_sec = float(times[start]) if times.size else 0.0
            last_time = float(times[min(end - 1, len(times) - 1)]) if times.size else start_sec
            end_sec = max(last_time + float(flow_sample_sec), start_sec + float(window_sec))
            tensors.append(window.astype(np.float32, copy=False))
            rows.append({
                "sample_id": sample_id,
                "video_id": sample_id,
                "target_label": target_label,
                "target_category": target_category,
                "target_environment": target_environment,
                "camera": str(camera),
                "window_start_sec": start_sec,
                "window_end_sec": float(end_sec),
                "window_center_sec": float(0.5 * (start_sec + end_sec)),
                "window_start_bin": int(round(start_sec / max(float(hop_sec), 1e-6))),
                "time": float(0.5 * (start_sec + end_sec)),
                "motion_feature_names": ",".join(motion_feature_names),
            })
    if not tensors:
        return np.zeros((0, time_steps, int(grid[0]), int(grid[1]), len(motion_feature_names)), dtype=np.float32), pd.DataFrame(rows)
    return np.stack(tensors, axis=0).astype(np.float32, copy=False), pd.DataFrame(rows)


def fit_score_calibration(scores: np.ndarray, quantiles: tuple[float, float] = (0.5, 0.995)) -> dict[str, float]:
    finite = np.asarray(scores, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"lower": 0.0, "upper": 1.0}
    lower_q, upper_q = sorted([float(quantiles[0]), float(quantiles[1])])
    lower = float(np.quantile(finite, np.clip(lower_q, 0.0, 1.0)))
    upper = float(np.quantile(finite, np.clip(upper_q, 0.0, 1.0)))
    if not np.isfinite(upper) or upper <= lower:
        upper = lower + 1e-6
    return {"lower": lower, "upper": upper}


def apply_score_calibration(scores: np.ndarray, calibration: dict[str, float]) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    lower = float(calibration.get("lower", 0.0))
    upper = float(calibration.get("upper", lower + 1.0))
    return np.clip((values - lower) / max(upper - lower, 1e-6), 0.0, 1.0)


def score_flow_tensor_windows(X: np.ndarray, model: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if X is None or len(X) == 0:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
    mean_tensor = np.asarray(model["mean_tensor"], dtype=np.float32)
    std_tensor = np.asarray(model["std_tensor"], dtype=np.float32)
    eps = float(model.get("eps", 1e-6))
    z = (np.asarray(X, dtype=np.float32) - mean_tensor) / np.maximum(std_tensor, eps)
    raw_scores = np.mean(z * z, axis=tuple(range(1, z.ndim))).astype(float)
    scores = apply_score_calibration(raw_scores, model.get("score_calibration", {"lower": 0.0, "upper": 1.0}))
    return raw_scores, scores


def explain_flow_tensor_windows(
    X: np.ndarray,
    model: dict[str, Any],
    meta_df: pd.DataFrame | None = None,
    *,
    prefix: str = "motion",
) -> pd.DataFrame:
    """Explain which camera/channel/grid contributed most to each motion score.

    The normal model scores windows with the mean squared z-score over
    ``(time, grid_y, grid_x, channel)``. This function exposes the largest
    component of that same contribution tensor so result tables can show whether
    a high motion score came mainly from front/rear, flow_x/flow_y, and which
    3x3 grid cell.
    """
    if X is None or len(X) == 0:
        return pd.DataFrame()
    values = np.asarray(X, dtype=np.float32)
    mean_tensor = np.asarray(model["mean_tensor"], dtype=np.float32)
    std_tensor = np.asarray(model["std_tensor"], dtype=np.float32)
    eps = float(model.get("eps", 1e-6))
    contributions = ((values - mean_tensor) / np.maximum(std_tensor, eps)) ** 2
    channels = list(model.get("channels", ["flow_x", "flow_y"]))
    if len(channels) < contributions.shape[-1]:
        channels.extend([f"channel_{idx}" for idx in range(len(channels), contributions.shape[-1])])
    channels = channels[:contributions.shape[-1]]

    meta = meta_df.reset_index(drop=True).copy() if meta_df is not None and len(meta_df) == len(values) else pd.DataFrame(index=range(len(values)))
    rows: list[dict[str, Any]] = []
    for index in range(len(values)):
        window_contrib = contributions[index]
        channel_contrib = window_contrib.mean(axis=(0, 1, 2))
        grid_channel_contrib = window_contrib.mean(axis=0)
        top_channel_index = int(np.nanargmax(channel_contrib)) if channel_contrib.size else 0
        gy, gx, grid_channel_index = np.unravel_index(
            int(np.nanargmax(grid_channel_contrib)),
            grid_channel_contrib.shape,
        )
        row = {
            f"{prefix}_top_camera": str(meta.at[index, "camera"]) if "camera" in meta.columns else "unknown",
            f"{prefix}_top_channel": str(channels[top_channel_index]),
            f"{prefix}_top_grid_row": int(gy),
            f"{prefix}_top_grid_col": int(gx),
            f"{prefix}_top_grid_label": f"{int(gx) + 1}x{int(gy) + 1}",
            f"{prefix}_top_grid_channel": str(channels[int(grid_channel_index)]),
            f"{prefix}_top_grid_contribution": float(grid_channel_contrib[int(gy), int(gx), int(grid_channel_index)]),
        }
        for channel_index, channel_name in enumerate(channels):
            row[f"{prefix}_{channel_name}_contribution"] = float(channel_contrib[channel_index]) if channel_index < channel_contrib.size else 0.0
        row.setdefault(f"{prefix}_flow_x_contribution", 0.0)
        row.setdefault(f"{prefix}_flow_y_contribution", 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def fit_normal_flow_tensor_model(
    X_train: np.ndarray,
    *,
    feature_names: Iterable[str] | None = None,
    calibration_quantiles: tuple[float, float] = (0.5, 0.995),
    eps: float = 1e-6,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if X_train is None or len(X_train) == 0:
        raise ValueError("X_train must contain at least one flow tensor window")
    X = np.asarray(X_train, dtype=np.float32)
    if feature_names is None and int(X.shape[-1]) == len(DEFAULT_MOTION_FEATURE_NAMES):
        channels = list(DEFAULT_MOTION_FEATURE_NAMES)
    else:
        channels = normalize_motion_feature_names(feature_names) if feature_names is not None else [f"channel_{idx}" for idx in range(int(X.shape[-1]))]
    if len(channels) != int(X.shape[-1]):
        raise ValueError(f"feature_names length must match tensor channel count: {len(channels)} != {int(X.shape[-1])}")
    mean_tensor = np.mean(X, axis=0).astype(np.float32)
    std_tensor = np.maximum(np.std(X, axis=0).astype(np.float32), float(eps))
    model = {
        "mean_tensor": mean_tensor,
        "std_tensor": std_tensor,
        "eps": float(eps),
        "tensor_shape": tuple(int(v) for v in mean_tensor.shape),
        "channels": list(channels),
    }
    raw_scores, _ = score_flow_tensor_windows(X, {**model, "score_calibration": {"lower": 0.0, "upper": 1.0}})
    model["score_calibration"] = fit_score_calibration(raw_scores, calibration_quantiles)
    _, scores = score_flow_tensor_windows(X, model)
    return model, raw_scores, scores


def aggregate_camera_scores(
    window_scores_df: pd.DataFrame,
    *,
    method: str = "max",
    score_col: str = "motion_anomaly_score",
    raw_score_col: str = "motion_anomaly_score_raw",
) -> pd.DataFrame:
    if window_scores_df is None or window_scores_df.empty:
        return pd.DataFrame()
    group_cols = ["sample_id", "window_start_bin", "window_start_sec", "window_end_sec", "window_center_sec"]
    available_group_cols = [col for col in group_cols if col in window_scores_df.columns]
    agg_func = "mean" if str(method).lower() == "mean" else "max"
    rows = []
    for keys, group in window_scores_df.groupby(available_group_cols, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(available_group_cols, keys))
        row[score_col] = float(getattr(group[score_col], agg_func)()) if score_col in group.columns else 0.0
        row[raw_score_col] = float(getattr(group[raw_score_col], agg_func)()) if raw_score_col in group.columns else 0.0
        row["camera_count"] = int(group["camera"].nunique()) if "camera" in group.columns else int(len(group))
        if score_col in group.columns and len(group):
            best_row = group.loc[group[score_col].astype(float).idxmax()]
            attribution_cols = [
                col for col in group.columns
                if col.startswith("motion_top_") or (col.startswith("motion_") and col.endswith("_contribution"))
            ]
            for col in attribution_cols:
                row[col] = best_row.get(col)
        if "target_label" in group.columns:
            row["target_label"] = str(group["target_label"].dropna().iat[0]) if group["target_label"].dropna().size else str(row.get("sample_id", "unknown"))
        if "target_category" in group.columns:
            row["target_category"] = str(group["target_category"].dropna().iat[0]) if group["target_category"].dropna().size else "unknown"
        if "target_environment" in group.columns:
            row["target_environment"] = str(group["target_environment"].dropna().iat[0]) if group["target_environment"].dropna().size else "unknown"
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["anomaly_score"] = out[score_col]
        out["anomaly_score_smooth"] = out.groupby("sample_id")["anomaly_score"].transform(lambda s: s.rolling(window=5, center=True, min_periods=1).mean())
    return out
