from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .broad_vibration import (
    DEFAULT_BROAD_VIBRATION_FEATURE_LABELS,
    build_broad_vibration_change_amount_column_name,
    build_broad_vibration_column_name,
)


DEFAULT_VIBRATION_SCORE_WEIGHTS = {
    "change_sum_z": 0.35,
    "change_high_ratio_z": 0.25,
    "change_variation_z": 0.25,
    "change_p95_z": 0.15,
}


def aggregate_feature_df(df: pd.DataFrame, window_sec: float) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "video_id" not in df.columns:
        df["video_id"] = "unknown"
    if "time_bin" not in df.columns:
        if "time" not in df.columns:
            raise ValueError("feature df must have either time_bin or time")
        df["time_bin"] = np.round(df["time"] / window_sec).astype(int)

    key_cols = ["video_id", "time_bin"]
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in {"time", "time_bin"}]
    other_cols = [c for c in df.columns if c not in set(key_cols + numeric_cols + ["time"])]
    out = df.groupby(key_cols, as_index=False)[numeric_cols].mean() if numeric_cols else df[key_cols].drop_duplicates()
    for col in other_cols:
        values = df.groupby(key_cols)[col].agg(lambda x: x.dropna().mode().iat[0] if len(x.dropna().mode()) else "unknown").reset_index()
        out = out.merge(values, on=key_cols, how="left")
    out["time"] = out["time_bin"] * window_sec
    return out


def align_features_by_time(feature_dfs: Sequence[pd.DataFrame], window_sec: float) -> pd.DataFrame:
    valid = [aggregate_feature_df(df, window_sec=window_sec) for df in feature_dfs if df is not None and len(df)]
    if not valid:
        return pd.DataFrame()
    merged = valid[0]
    for df in valid[1:]:
        merged = merged.merge(df.drop(columns=["time"], errors="ignore"), on=["video_id", "time_bin"], how="outer")
    merged["time"] = merged["time_bin"] * window_sec
    merged = merged.sort_values(["video_id", "time_bin"]).reset_index(drop=True)
    numeric_cols = [c for c in merged.select_dtypes(include=[np.number]).columns if c not in {"time", "time_bin"}]
    merged[numeric_cols] = merged.groupby("video_id", group_keys=False)[numeric_cols].apply(lambda g: g.ffill().bfill()).fillna(0.0)
    for col in merged.select_dtypes(exclude=[np.number]).columns:
        if col != "video_id":
            merged[col] = merged[col].fillna("unknown")
    return merged


def robust_positive_zscore(values: np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.zeros((0,), dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=float)
    median = float(np.median(finite))
    scale = max(1.4826 * float(np.median(np.abs(finite - median))), 1e-6)
    return np.maximum((np.nan_to_num(values, nan=median) - median) / scale, 0.0)


def percentile_normalize_0_1(
    values: np.ndarray | pd.Series,
    lower_percentile: float = 0.0,
    upper_percentile: float = 95.0,
    *,
    positive_only: bool = True,
    min_visible: float = 1e-6,
) -> np.ndarray:
    values_arr = np.asarray(values, dtype=float)
    out = np.zeros_like(values_arr, dtype=float)
    finite_mask = np.isfinite(values_arr)
    if not finite_mask.any():
        return out
    fit_values = values_arr[finite_mask]
    if positive_only:
        fit_values = fit_values[fit_values > min_visible]
    if fit_values.size == 0:
        return out
    lo_q, hi_q = sorted([float(np.clip(lower_percentile, 0.0, 100.0)), float(np.clip(upper_percentile, 0.0, 100.0))])
    lo, hi = float(np.percentile(fit_values, lo_q)), float(np.percentile(fit_values, hi_q))
    if hi <= lo:
        out[finite_mask] = np.where(values_arr[finite_mask] > min_visible, 1.0, 0.0)
    else:
        out[finite_mask] = np.clip((values_arr[finite_mask] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return np.where(values_arr > min_visible, out, 0.0).astype(float) if positive_only else out.astype(float)


def build_flow_window_starts(duration_sec: float, window_sec: float, hop_sec: float) -> list[float]:
    max_start = max(float(duration_sec) - float(window_sec), 0.0)
    starts = np.arange(0.0, max_start + 1e-9, float(hop_sec), dtype=np.float32).tolist()
    return [float(value) for value in (starts if starts else [0.0])]


def _reliable_ratio_values(ordered: pd.DataFrame, x_col: str, reliable_col: str | None, size: int) -> np.ndarray:
    candidates = [
        reliable_col,
        x_col.replace("_x_mean", "_flow_reliable_ratio"),
        x_col.replace("_x_mean", "_valid_ratio"),
    ]
    for candidate in candidates:
        if candidate and candidate in ordered.columns:
            return pd.to_numeric(ordered[candidate], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    return np.zeros(size, dtype=float)


def compute_flow_change_metric(raw_flow_df: pd.DataFrame, x_col: str, y_col: str, feature_name: str, reliable_col: str | None = None) -> pd.Series:
    if x_col not in raw_flow_df.columns or y_col not in raw_flow_df.columns or "time" not in raw_flow_df.columns:
        return pd.Series(dtype=float)
    ordered = raw_flow_df.sort_values("time")
    x = ordered[x_col].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    y = ordered[y_col].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    prev_x = np.r_[x[:1], x[:-1]] if x.size else x
    prev_y = np.r_[y[:1], y[:-1]] if y.size else y

    if feature_name == "vector_change":
        values = np.hypot(x - prev_x, y - prev_y)
    elif feature_name == "flow_reliable_ratio":
        values = _reliable_ratio_values(ordered, x_col, reliable_col, x.size)
    elif feature_name in {"reliability_weighted_vector_change", "reliable_vector_change"}:
        reliable_ratio = _reliable_ratio_values(ordered, x_col, reliable_col, x.size)
        values = np.hypot(x - prev_x, y - prev_y) * (reliable_ratio ** 2)
    elif feature_name == "x_change":
        values = np.abs(x - prev_x)
    elif feature_name == "y_change":
        values = np.abs(y - prev_y)
    else:
        raise ValueError(f"Unknown broad vibration feature: {feature_name}")

    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.Series(values, index=ordered.index).reindex(raw_flow_df.index).fillna(0.0)


def compute_grid_flow_change_values(grid_part: pd.DataFrame, feature_name: str, direction_min_mag: float = 0.025) -> pd.Series:
    ordered = grid_part.sort_values("time")
    x = ordered["flow_x"].astype(float).to_numpy()
    y = ordered["flow_y"].astype(float).to_numpy()
    prev_x = np.r_[x[:1], x[:-1]] if x.size else x
    prev_y = np.r_[y[:1], y[:-1]] if y.size else y
    magnitude = np.hypot(x, y)
    prev_magnitude = np.hypot(prev_x, prev_y)
    if feature_name == "direction_change":
        angle = np.arctan2(y, x)
        prev_angle = np.r_[angle[:1], angle[:-1]] if angle.size else angle
        delta = angle - prev_angle
        values = np.abs(np.arctan2(np.sin(delta), np.cos(delta)))
        valid = (magnitude >= direction_min_mag) & (prev_magnitude >= direction_min_mag)
        values = np.where(valid, np.clip(values, 0.0, np.pi), 0.0)
    elif feature_name == "magnitude_change":
        values = np.abs(magnitude - prev_magnitude)
    elif feature_name == "vector_change":
        values = np.hypot(x - prev_x, y - prev_y)
    elif feature_name in {"reliable_vector_change", "reliability_weighted_vector_change"}:
        reliable_ratio = pd.to_numeric(ordered["flow_reliable_ratio"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        values = np.hypot(x - prev_x, y - prev_y) * (reliable_ratio ** 2)
    elif feature_name == "flow_reliable_ratio":
        values = pd.to_numeric(ordered["flow_reliable_ratio"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    elif feature_name == "change_vector_direction_change":
        change_x = x - prev_x
        change_y = y - prev_y
        change_magnitude = np.hypot(change_x, change_y)
        change_angle = np.degrees(np.arctan2(change_y, change_x))
        change_angle = np.where(change_magnitude >= direction_min_mag, change_angle, 0.0)
        prev_change_angle = np.r_[0.0, change_angle[:-1]] if change_angle.size else change_angle
        values = np.abs((change_angle - prev_change_angle + 180.0) % 360.0 - 180.0)
    elif feature_name == "x_change":
        values = np.abs(x - prev_x)
    elif feature_name == "y_change":
        values = np.abs(y - prev_y)
    else:
        raise ValueError(f"Unknown broad vibration feature: {feature_name}")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.Series(values, index=ordered.index).reindex(grid_part.index).fillna(0.0)


def mark_vibration_hysteresis_windows(window_df: pd.DataFrame, score_col: str = "vibration_score") -> pd.DataFrame:
    window_df = window_df.sort_values("window_start_sec").copy()
    scores = pd.to_numeric(window_df[score_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    active = np.ones(len(window_df), dtype=bool)
    window_df["vibration_seed"] = active
    window_df["vibration_selected"] = active
    window_df["vibration_high_threshold"] = np.nan
    window_df["vibration_low_threshold"] = np.nan
    window_df["thresholded_vibration_score"] = scores.astype(float)
    return window_df


def add_vibration_score_alpha(
    window_df: pd.DataFrame,
    *,
    alpha_min: float = 0.04,
    alpha_max: float = 0.42,
    min_visible: float = 1e-6,
    mark_windows: bool = False,
) -> pd.DataFrame:
    if window_df.empty or "vibration_score" not in window_df.columns:
        work = window_df.copy()
        if mark_windows:
            work["vibration_seed"] = False
            work["vibration_selected"] = False
            work["thresholded_vibration_score"] = 0.0
        work["vibration_score_alpha"] = 0.0
        return work
    work = mark_vibration_hysteresis_windows(window_df) if mark_windows else window_df.copy()
    scores = pd.to_numeric(work["vibration_score"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    finite = scores[np.isfinite(scores)]
    if finite.size == 0 or float(np.nanmax(finite)) <= min_visible:
        work["vibration_score_alpha"] = 0.0
        return work
    lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    normalized = np.where(scores > min_visible, 1.0, 0.0) if hi <= lo else np.clip((scores - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    work["vibration_score_alpha"] = np.where(scores > min_visible, alpha_min + normalized * (alpha_max - alpha_min), 0.0).astype(float)
    return work


def _add_vibration_score_columns(
    window_df: pd.DataFrame,
    *,
    lower_percentile: float,
    upper_percentile: float,
    min_visible: float,
) -> pd.DataFrame:
    for name in ["change_sum", "change_high_ratio", "change_variation", "change_p95"]:
        window_df[f"{name}_z"] = robust_positive_zscore(window_df[name])
    window_df["vibration_score_raw"] = sum(
        weight * window_df[column]
        for column, weight in DEFAULT_VIBRATION_SCORE_WEIGHTS.items()
    ).astype(float)
    window_df["vibration_score"] = percentile_normalize_0_1(
        window_df["vibration_score_raw"],
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        min_visible=min_visible,
    )
    return window_df


def build_flow_vibration_window_table(
    raw_flow_df: pd.DataFrame,
    camera: str,
    gy: int,
    gx: int,
    feature_name: str,
    *,
    flow_score_window_sec: float,
    flow_score_hop_sec: float,
    flow_score_min_visible: float,
    flow_score_high_ratio_fraction: float,
    vibration_score_lower_percentile: float,
    vibration_score_upper_percentile: float,
    feature_labels: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    feature_labels = feature_labels or DEFAULT_BROAD_VIBRATION_FEATURE_LABELS
    x_col = f"{camera}_flow_cell_{gy}_{gx}_x_mean"
    y_col = f"{camera}_flow_cell_{gy}_{gx}_y_mean"
    reliable_col = f"{camera}_flow_cell_{gy}_{gx}_flow_reliable_ratio"
    if reliable_col not in raw_flow_df.columns:
        reliable_col = f"{camera}_flow_cell_{gy}_{gx}_valid_ratio"
    if raw_flow_df.empty or x_col not in raw_flow_df.columns or y_col not in raw_flow_df.columns or "time" not in raw_flow_df.columns:
        return pd.DataFrame()

    ordered = raw_flow_df.sort_values("time").reset_index(drop=True)
    change_values = compute_flow_change_metric(ordered, x_col, y_col, feature_name, reliable_col=reliable_col).to_numpy(dtype=float)
    times = ordered["time"].astype(float).to_numpy()
    if times.size == 0:
        return pd.DataFrame()
    duration_sec = max(float(np.nanmax(times)) if np.isfinite(times).any() else 0.0, flow_score_window_sec)

    rows = []
    for start_sec in build_flow_window_starts(duration_sec, flow_score_window_sec, flow_score_hop_sec):
        end_sec = float(min(start_sec + flow_score_window_sec, duration_sec))
        start_sec = max(end_sec - flow_score_window_sec, 0.0)
        mask = (times >= start_sec) & (times < end_sec)
        if not np.any(mask):
            mask = np.zeros_like(times, dtype=bool)
            mask[int(np.argmin(np.abs(times - start_sec)))] = True
        segment = np.nan_to_num(change_values[mask], nan=0.0, posinf=0.0, neginf=0.0)
        change_max = float(np.max(segment)) if segment.size else 0.0
        high_ratio = float(np.mean(segment >= change_max * flow_score_high_ratio_fraction)) if change_max > flow_score_min_visible else 0.0
        rows.append({
            "camera": camera,
            "grid_x": gx + 1,
            "grid_y": gy + 1,
            "grid_col": gx,
            "grid_row": gy,
            "vibration_feature": feature_name,
            "vibration_feature_label": feature_labels.get(feature_name, feature_name),
            "window_start_sec": float(start_sec),
            "window_end_sec": float(end_sec),
            "window_center_sec": float(0.5 * (start_sec + end_sec)),
            "change_mean": float(np.mean(segment)) if segment.size else 0.0,
            "change_sum": float(np.sum(segment)) if segment.size else 0.0,
            "change_p95": float(np.percentile(segment, 95)) if segment.size else 0.0,
            "change_max": change_max,
            "change_std": float(np.std(segment)) if segment.size else 0.0,
            "change_variation": float(np.mean(np.abs(np.diff(segment)))) if segment.size >= 2 else 0.0,
            "change_high_ratio": high_ratio,
        })

    window_df = pd.DataFrame(rows)
    return _add_vibration_score_columns(
        window_df,
        lower_percentile=vibration_score_lower_percentile,
        upper_percentile=vibration_score_upper_percentile,
        min_visible=flow_score_min_visible,
    )


def build_flow_vibration_score_table(
    raw_flow_df: pd.DataFrame,
    camera: str,
    feature_name: str,
    *,
    flow_grid: tuple[int, int],
    flow_score_window_sec: float,
    flow_score_hop_sec: float,
    flow_score_min_visible: float,
    flow_score_high_ratio_fraction: float,
    vibration_score_lower_percentile: float,
    vibration_score_upper_percentile: float,
    feature_labels: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    tables = [
        build_flow_vibration_window_table(
            raw_flow_df,
            camera,
            gy,
            gx,
            feature_name,
            flow_score_window_sec=flow_score_window_sec,
            flow_score_hop_sec=flow_score_hop_sec,
            flow_score_min_visible=flow_score_min_visible,
            flow_score_high_ratio_fraction=flow_score_high_ratio_fraction,
            vibration_score_lower_percentile=vibration_score_lower_percentile,
            vibration_score_upper_percentile=vibration_score_upper_percentile,
            feature_labels=feature_labels,
        )
        for gy in range(flow_grid[0])
        for gx in range(flow_grid[1])
    ]
    tables = [df for df in tables if len(df)]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def normalize_vibration_score_table(
    vibration_df: pd.DataFrame,
    *,
    flow_score_hop_sec: float,
    selected_default: bool = True,
) -> pd.DataFrame:
    work = vibration_df.copy()
    work["window_start_bin"] = np.round(work["window_start_sec"].astype(float) / max(flow_score_hop_sec, 1e-6)).astype(int)
    if selected_default and "vibration_selected" not in work.columns:
        work["vibration_selected"] = True
    return work


def add_vibration_topk_columns(vibration_df: pd.DataFrame, *, flow_score_hop_sec: float, top_k: int | None = None) -> pd.DataFrame:
    if vibration_df is None or vibration_df.empty:
        return pd.DataFrame()
    work = normalize_vibration_score_table(vibration_df, flow_score_hop_sec=flow_score_hop_sec)
    required = {"video_id", "camera", "window_start_sec", "vibration_feature", "vibration_score"}
    missing = sorted(required - set(work.columns))
    if missing:
        raise ValueError(f"Missing columns for vibration grid averaging: {missing}")

    scores = pd.to_numeric(work["vibration_score"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    work["thresholded_vibration_score"] = scores.astype(float)
    work["vibration_topk_rank"] = pd.Series(pd.NA, index=work.index, dtype="Int64")
    work["vibration_topk_selected"] = True

    meta_cols = [c for c in ["target_label", "target_category", "target_environment"] if c in work.columns]
    for _, group in work.groupby(["video_id", *meta_cols, "vibration_feature", "camera", "window_start_bin"], sort=False):
        ranked = group.sort_values("thresholded_vibration_score", ascending=False)
        if len(ranked):
            work.loc[ranked.index, "vibration_topk_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    return work


def build_broad_vibration_score_table(
    vibration_df: pd.DataFrame,
    *,
    id_col: str = "video_id",
    camera_col: str = "camera",
    flow_score_hop_sec: float = 0.5,
    feature_labels: Mapping[str, str] | None = None,
    top_k: int | None = None,
) -> pd.DataFrame:
    if vibration_df is None or vibration_df.empty:
        return pd.DataFrame()
    feature_labels = feature_labels or DEFAULT_BROAD_VIBRATION_FEATURE_LABELS
    work = normalize_vibration_score_table(vibration_df, flow_score_hop_sec=flow_score_hop_sec)
    meta_cols = [c for c in ["target_label", "target_category", "target_environment"] if c in work.columns]
    rows = []
    for keys, group in work.groupby([id_col, *meta_cols, "vibration_feature", camera_col, "window_start_bin"], sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        item_id = keys[0]
        meta_values = dict(zip(meta_cols, keys[1:1 + len(meta_cols)]))
        feature_name = keys[1 + len(meta_cols)]
        camera = keys[2 + len(meta_cols)]
        window_start_bin = keys[3 + len(meta_cols)]
        scores = pd.to_numeric(group["vibration_score"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        change_sum_series = group["change_sum"] if "change_sum" in group.columns else pd.Series(0.0, index=group.index)
        change_sums = pd.to_numeric(change_sum_series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        broad_score = float(np.mean(scores)) if scores.size else 0.0
        change_amount_scores = ((broad_score ** 2) * change_sums).astype(float)
        rows.append({
            id_col: item_id,
            **meta_values,
            "broad_vibration_feature": feature_name,
            "broad_vibration_feature_label": feature_labels.get(str(feature_name), str(feature_name)),
            camera_col: camera,
            "window_start_bin": int(window_start_bin),
            "window_start_sec": float(group["window_start_sec"].min()),
            "window_end_sec": float(group["window_end_sec"].max()),
            "window_center_sec": float(group["window_center_sec"].median()),
            "broad_vibration_score": broad_score,
            "broad_vibration_change_amount_score": float(np.mean(change_amount_scores)) if change_amount_scores.size else 0.0,
            "broad_vibration_grid_count": int(len(group)),
            "mean_change_sum": float(np.mean(change_sums)) if change_sums.size else 0.0,
            "max_vibration_score": float(np.max(scores)) if scores.size else 0.0,
            "max_vibration_change_amount_score": float(np.max(change_amount_scores)) if change_amount_scores.size else 0.0,
        })
    return pd.DataFrame(rows).sort_values([id_col, "broad_vibration_feature", camera_col, "window_start_sec"]).reset_index(drop=True)


def build_broad_vibration_feature_df(
    raw_flow_df: pd.DataFrame,
    *,
    broad_vibration_columns: Sequence[str],
    broad_vibration_features: Mapping[str, Any],
    flow_grid: tuple[int, int],
    window_sec: float,
    flow_score_window_sec: float,
    flow_score_hop_sec: float,
    flow_score_min_visible: float,
    flow_score_high_ratio_fraction: float,
    vibration_score_lower_percentile: float,
    vibration_score_upper_percentile: float,
    feature_labels: Mapping[str, str] | None = None,
    use_front: bool = True,
    use_rear: bool = True,
    top_k: int | None = None,
) -> pd.DataFrame:
    empty_cols = ["video_id", "time", "time_bin", *broad_vibration_columns]
    if raw_flow_df is None or raw_flow_df.empty:
        return pd.DataFrame(columns=empty_cols)

    raw_flow_df = raw_flow_df.copy()
    if "video_id" not in raw_flow_df.columns:
        raw_flow_df["video_id"] = "unknown"
    vibration_tables = []
    cameras = [("front", use_front), ("rear", use_rear)]
    for video_id, video_df in raw_flow_df.groupby("video_id", sort=False):
        meta = {c: video_df[c].dropna().iloc[0] for c in ["target_label", "target_category", "target_environment"] if c in video_df.columns and video_df[c].notna().any()}
        for camera, enabled in cameras:
            if not enabled:
                continue
            for feature_name in broad_vibration_features:
                vibration_df = build_flow_vibration_score_table(
                    video_df,
                    camera,
                    feature_name,
                    flow_grid=flow_grid,
                    flow_score_window_sec=flow_score_window_sec,
                    flow_score_hop_sec=flow_score_hop_sec,
                    flow_score_min_visible=flow_score_min_visible,
                    flow_score_high_ratio_fraction=flow_score_high_ratio_fraction,
                    vibration_score_lower_percentile=vibration_score_lower_percentile,
                    vibration_score_upper_percentile=vibration_score_upper_percentile,
                    feature_labels=feature_labels,
                )
                if len(vibration_df):
                    vibration_df.insert(0, "video_id", video_id)
                    for col, value in meta.items():
                        vibration_df[col] = value
                    vibration_tables.append(vibration_df)
    if not vibration_tables:
        return pd.DataFrame(columns=empty_cols)

    broad_df = build_broad_vibration_score_table(
        pd.concat(vibration_tables, ignore_index=True),
        id_col="video_id",
        camera_col="camera",
        flow_score_hop_sec=flow_score_hop_sec,
        feature_labels=feature_labels,
        top_k=top_k,
    )
    if broad_df.empty:
        return pd.DataFrame(columns=empty_cols)

    broad_df = broad_df.copy()
    broad_df["time"] = broad_df["window_start_sec"].astype(float)
    broad_df["time_bin"] = np.round(broad_df["time"] / max(window_sec, 1e-6)).astype(int)

    def pivot_broad_value(value_col: str, column_builder) -> pd.DataFrame:
        pivot = broad_df.pivot_table(
            index=["video_id", "time_bin", "time"],
            columns=["camera", "broad_vibration_feature"],
            values=value_col,
            aggfunc="mean",
        ).reset_index()
        flat_columns = []
        for col in pivot.columns:
            if isinstance(col, tuple):
                if len(col) >= 2 and col[1] != "":
                    flat_columns.append(column_builder(str(col[0]), str(col[1])))
                else:
                    flat_columns.append(str(col[0]))
            else:
                flat_columns.append(str(col))
        out = pivot.copy()
        out.columns = flat_columns
        return out

    score_df = pivot_broad_value("broad_vibration_score", build_broad_vibration_column_name)
    change_amount_df = pivot_broad_value("broad_vibration_change_amount_score", build_broad_vibration_change_amount_column_name)
    feature_df = score_df.merge(change_amount_df.drop(columns=["time"], errors="ignore"), on=["video_id", "time_bin"], how="outer")
    feature_df["time"] = feature_df["time"].fillna(feature_df["time_bin"] * window_sec)
    for col in broad_vibration_columns:
        if col not in feature_df.columns:
            feature_df[col] = 0.0
    return feature_df[empty_cols].copy()


def ensure_broad_vibration_columns(df: pd.DataFrame, broad_vibration_columns: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    for col in broad_vibration_columns:
        if col not in df.columns:
            df[col] = 0.0
        else:
            values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            df[col] = values.groupby(df["video_id"]).transform(lambda s: s.ffill().bfill()).fillna(0.0) if "video_id" in df.columns else values.fillna(0.0)
    return df


def build_grid_vibration_score_table(
    grid_df: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    window_sec: float = 1.0,
    hop_sec: float = 0.5,
    alpha_min: float = 0.04,
    alpha_max: float = 0.42,
    feature_labels: Mapping[str, str] | None = None,
    score_min_visible: float = 1e-6,
    high_ratio_fraction: float = 0.5,
    lower_percentile: float = 0.0,
    upper_percentile: float = 95.0,
) -> pd.DataFrame:
    if grid_df.empty:
        return pd.DataFrame()
    feature_labels = feature_labels or DEFAULT_BROAD_VIBRATION_FEATURE_LABELS
    rows = []
    group_cols = ["sample_id", "view", "grid_row", "grid_col"]
    for keys, grid_part in grid_df.groupby(group_cols, sort=False):
        sample_id, view, grid_row, grid_col = keys
        grid_part = grid_part.sort_values("time")
        times = grid_part["time"].astype(float).to_numpy()
        if times.size == 0:
            continue
        duration_sec = max(float(np.nanmax(times)) if np.isfinite(times).any() else 0.0, window_sec)
        for feature_name in feature_names:
            change_values = compute_grid_flow_change_values(grid_part, feature_name).to_numpy(dtype=float)
            feature_rows = []
            for start_sec in build_flow_window_starts(duration_sec, window_sec, hop_sec):
                end_sec = float(min(start_sec + window_sec, duration_sec))
                start_sec = max(end_sec - window_sec, 0.0)
                mask = (times >= start_sec) & (times < end_sec)
                if not np.any(mask):
                    mask = np.zeros_like(times, dtype=bool)
                    mask[int(np.argmin(np.abs(times - start_sec)))] = True
                segment = np.nan_to_num(change_values[mask], nan=0.0, posinf=0.0, neginf=0.0)
                change_max = float(np.max(segment)) if segment.size else 0.0
                high_ratio = float(np.mean(segment >= change_max * high_ratio_fraction)) if change_max > score_min_visible else 0.0
                feature_rows.append({
                    "sample_id": sample_id,
                    "view": view,
                    "grid_row": int(grid_row),
                    "grid_col": int(grid_col),
                    "vibration_feature": feature_name,
                    "vibration_feature_label": feature_labels.get(feature_name, feature_name),
                    "window_start_sec": float(start_sec),
                    "window_end_sec": float(end_sec),
                    "window_center_sec": float(0.5 * (start_sec + end_sec)),
                    "change_mean": float(np.mean(segment)) if segment.size else 0.0,
                    "change_sum": float(np.sum(segment)) if segment.size else 0.0,
                    "change_p95": float(np.percentile(segment, 95)) if segment.size else 0.0,
                    "change_max": change_max,
                    "change_std": float(np.std(segment)) if segment.size else 0.0,
                    "change_variation": float(np.mean(np.abs(np.diff(segment)))) if segment.size >= 2 else 0.0,
                    "change_high_ratio": high_ratio,
                })
            feature_df = pd.DataFrame(feature_rows)
            if feature_df.empty:
                continue
            rows.append(_add_vibration_score_columns(
                feature_df,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
                min_visible=score_min_visible,
            ))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return add_vibration_score_alpha(out, alpha_min=alpha_min, alpha_max=alpha_max, min_visible=score_min_visible) if len(out) else out


def build_grid_broad_vibration_score_table(
    vibration_df: pd.DataFrame,
    *,
    hop_sec: float = 0.5,
    feature_labels: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    return build_broad_vibration_score_table(
        vibration_df,
        id_col="sample_id",
        camera_col="view",
        flow_score_hop_sec=hop_sec,
        feature_labels=feature_labels,
    )
