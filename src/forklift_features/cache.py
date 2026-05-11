from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd


DEFAULT_FEATURE_CACHE_VERSION = "sample_feature_cache_v7"
DEFAULT_FLOW_CACHE_VERSION = "sample_flow_cache_v13"
DEFAULT_FLOW_CACHE_METHOD = "farneback_event_update_interval_apportioned_0p1s"


def normalize_cache_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(k): normalize_cache_value(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [normalize_cache_value(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def is_missing_path_value(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def file_cache_fingerprint(value: Any) -> dict[str, Any]:
    if is_missing_path_value(value):
        return {"path": None, "exists": False}
    path = Path(value)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def build_feature_cache_settings(
    *,
    cache_version: str = DEFAULT_FEATURE_CACHE_VERSION,
    use_front: bool = True,
    use_rear: bool = True,
    flow_sample_sec: float,
    window_sec: float,
    audio_sr: int,
    n_mels: int,
    frame_resize_width: int | None,
    flow_analysis_scale: float,
    flow_grid: tuple[int, int],
    flow_reliable_error_threshold_px: float,
    flow_score_window_sec: float,
    flow_score_hop_sec: float,
    flow_score_min_visible: float,
    flow_score_high_ratio_fraction: float,
    vibration_score_lower_percentile: float,
    vibration_score_upper_percentile: float,
    broad_vibration_feature_names: list[str],
    broad_vibration_top_k: int | None = None,
    broad_vibration_spread_power: float = 1.0,
    flow_method: str = DEFAULT_FLOW_CACHE_METHOD,
) -> dict[str, Any]:
    return normalize_cache_value({
        "cache_version": cache_version,
        "use_front": use_front,
        "use_rear": use_rear,
        "flow_sample_sec": flow_sample_sec,
        "window_sec": window_sec,
        "audio_sr": audio_sr,
        "n_mels": n_mels,
        "frame_resize_width": frame_resize_width,
        "flow_analysis_scale": flow_analysis_scale,
        "flow_grid": flow_grid,
        "flow_reliable_error_threshold_px": flow_reliable_error_threshold_px,
        "flow_method": flow_method,
        "flow_score_window_sec": flow_score_window_sec,
        "flow_score_hop_sec": flow_score_hop_sec,
        "flow_score_min_visible": flow_score_min_visible,
        "flow_score_high_ratio_fraction": flow_score_high_ratio_fraction,
        "vibration_score_lower_percentile": vibration_score_lower_percentile,
        "vibration_score_upper_percentile": vibration_score_upper_percentile,
        "broad_vibration_top_k": broad_vibration_top_k,
        "broad_vibration_spread_power": broad_vibration_spread_power,
        "broad_vibration_feature_names": list(broad_vibration_feature_names),
    })


def build_sample_flow_cache_settings(
    *,
    cache_version: str = DEFAULT_FLOW_CACHE_VERSION,
    use_front: bool = True,
    use_rear: bool = True,
    flow_sample_sec: float,
    frame_resize_width: int | None,
    flow_analysis_scale: float,
    flow_grid: tuple[int, int],
    flow_reliable_error_threshold_px: float,
    flow_method: str = DEFAULT_FLOW_CACHE_METHOD,
) -> dict[str, Any]:
    """Settings that affect raw optical-flow measurement only.

    This cache is intentionally separate from the full sample feature cache: a
    visualization notebook can save raw flow without pretending that audio/model
    features have also been computed.
    """
    return normalize_cache_value({
        "cache_version": cache_version,
        "use_front": use_front,
        "use_rear": use_rear,
        "flow_sample_sec": flow_sample_sec,
        "frame_resize_width": frame_resize_width,
        "flow_analysis_scale": flow_analysis_scale,
        "flow_grid": flow_grid,
        "flow_reliable_error_threshold_px": flow_reliable_error_threshold_px,
        "flow_method": flow_method,
    })


def build_sample_feature_cache_metadata(sample: pd.Series | dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    getter = sample.get
    return {
        "sample_id": str(getter("sample_id")),
        "files": {
            "audio": file_cache_fingerprint(getter("audio_path")),
            "front": file_cache_fingerprint(getter("front_path")),
            "rear": file_cache_fingerprint(getter("rear_path")),
            "front_frame_map": file_cache_fingerprint(getter("front_frame_map_path")),
            "rear_frame_map": file_cache_fingerprint(getter("rear_frame_map_path")),
        },
        "settings": normalize_cache_value(settings),
    }


def build_sample_flow_cache_metadata(sample: pd.Series | dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    getter = sample.get
    return {
        "sample_id": str(getter("sample_id")),
        "files": {
            "front": file_cache_fingerprint(getter("front_path")),
            "rear": file_cache_fingerprint(getter("rear_path")),
            "front_frame_map": file_cache_fingerprint(getter("front_frame_map_path")),
            "rear_frame_map": file_cache_fingerprint(getter("rear_frame_map_path")),
        },
        "settings": normalize_cache_value(settings),
    }


def sample_feature_cache_path(
    sample: pd.Series | dict[str, Any],
    settings: dict[str, Any],
    cache_dir: str | Path,
) -> tuple[Path, dict[str, Any], str]:
    metadata = build_sample_feature_cache_metadata(sample, settings)
    metadata_json = json.dumps(metadata, sort_keys=True, ensure_ascii=True, default=str)
    cache_hash = hashlib.sha256(metadata_json.encode("utf-8")).hexdigest()[:20]
    sample_id = str(sample.get("sample_id"))
    safe_sample_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in sample_id)[:80]
    return Path(cache_dir) / f"{safe_sample_id}_{cache_hash}.joblib", metadata, cache_hash


def sample_flow_cache_path(
    sample: pd.Series | dict[str, Any],
    settings: dict[str, Any],
    cache_dir: str | Path,
) -> tuple[Path, dict[str, Any], str]:
    metadata = build_sample_flow_cache_metadata(sample, settings)
    metadata_json = json.dumps(metadata, sort_keys=True, ensure_ascii=True, default=str)
    cache_hash = hashlib.sha256(metadata_json.encode("utf-8")).hexdigest()[:20]
    sample_id = str(sample.get("sample_id"))
    safe_sample_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in sample_id)[:80]
    return Path(cache_dir) / "flow" / f"{safe_sample_id}_{cache_hash}.joblib", metadata, cache_hash


def load_sample_feature_cache(cache_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cached = joblib.load(cache_path)
    if isinstance(cached, dict) and "features" in cached:
        features_df = cached["features"]
        raw_flow_df = cached.get("raw_flow", pd.DataFrame())
        metadata = cached.get("metadata", {})
        return features_df, raw_flow_df, metadata
    return cached, pd.DataFrame(), {}


def save_sample_feature_cache(
    cache_path: str | Path,
    metadata: dict[str, Any],
    features_df: pd.DataFrame,
    raw_flow_df: pd.DataFrame | None = None,
    *,
    compress: int = 3,
) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    joblib.dump({"metadata": metadata, "features": features_df, "raw_flow": raw_flow_df if raw_flow_df is not None else pd.DataFrame()}, tmp_path, compress=compress)
    tmp_path.replace(cache_path)


def load_sample_flow_cache(cache_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    cached = joblib.load(cache_path)
    if isinstance(cached, dict) and "raw_flow" in cached:
        return cached.get("raw_flow", pd.DataFrame()), cached.get("metadata", {})
    if isinstance(cached, pd.DataFrame):
        return cached, {}
    return pd.DataFrame(), {}


def save_sample_flow_cache(
    cache_path: str | Path,
    metadata: dict[str, Any],
    raw_flow_df: pd.DataFrame | None,
    *,
    compress: int = 3,
) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    joblib.dump({"metadata": metadata, "raw_flow": raw_flow_df if raw_flow_df is not None else pd.DataFrame()}, tmp_path, compress=compress)
    tmp_path.replace(cache_path)


def apply_sample_metadata_to_features(df: pd.DataFrame, sample: pd.Series | dict[str, Any], *, label_default: str = "normal") -> pd.DataFrame:
    df = df.copy()
    df["environment"] = sample.get("environment", "unknown")
    df["label"] = sample.get("category", label_default)
    if "category" in sample:
        df["target_category"] = sample.get("category", "unknown")
    if "environment" in sample:
        df["target_environment"] = sample.get("environment", "unknown")
    if "plot_label" in sample or "sample_id" in sample:
        df["target_label"] = sample.get("plot_label", str(sample.get("sample_id", "unknown")))
    return df


def apply_sample_metadata_to_raw_flow(raw_flow_df: pd.DataFrame, sample: pd.Series | dict[str, Any]) -> pd.DataFrame:
    raw_flow_df = raw_flow_df.copy()
    if raw_flow_df.empty and not {"video_id", "time", "time_bin"}.issubset(raw_flow_df.columns):
        raw_flow_df = pd.DataFrame(columns=["video_id", "time", "time_bin"])
    if "category" in sample:
        raw_flow_df["target_category"] = sample.get("category", "unknown")
    if "environment" in sample:
        raw_flow_df["target_environment"] = sample.get("environment", "unknown")
    raw_flow_df["target_label"] = sample.get("plot_label", str(sample.get("sample_id", "unknown")))
    return raw_flow_df


def load_or_extract_sample_feature_cache(
    sample: pd.Series | dict[str, Any],
    *,
    settings: dict[str, Any],
    cache_dir: str | Path,
    extract_func,
    enable_cache: bool = True,
    label_default: str = "normal",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cache_path, metadata, cache_hash = sample_feature_cache_path(sample, settings, cache_dir)
    cache_info = {
        "sample_id": str(sample.get("sample_id")),
        "cache_key": cache_hash,
        "cache_path": str(cache_path),
        "cache_status": "disabled" if not enable_cache else "miss",
        "n_rows": 0,
    }
    if enable_cache and cache_path.exists():
        try:
            features_df, raw_flow_df, _ = load_sample_feature_cache(cache_path)
            features_df = apply_sample_metadata_to_features(features_df, sample, label_default=label_default)
            raw_flow_df = apply_sample_metadata_to_raw_flow(raw_flow_df, sample)
            cache_info["cache_status"] = "hit" if len(raw_flow_df) else "hit_features_only"
            cache_info["n_rows"] = int(len(features_df))
            return features_df, raw_flow_df, cache_info
        except Exception as exc:
            warnings.warn(f"feature cache load failed for sample_id={sample.get('sample_id')}: {exc}. Recomputing.")
            cache_info["cache_status"] = "load_failed"

    extracted = extract_func(pd.Series(sample), return_raw_flow=True)
    features_df, raw_flow_df = extracted
    features_df = apply_sample_metadata_to_features(features_df, sample, label_default=label_default)
    raw_flow_df = apply_sample_metadata_to_raw_flow(raw_flow_df, sample)
    cache_info["n_rows"] = int(len(features_df))
    if enable_cache:
        save_sample_feature_cache(cache_path, metadata, features_df, raw_flow_df)
        cache_info["cache_status"] = "saved" if cache_info["cache_status"] == "miss" else f"{cache_info['cache_status']}_saved"
    return features_df, raw_flow_df, cache_info


def load_or_extract_sample_flow_cache(
    sample: pd.Series | dict[str, Any],
    *,
    settings: dict[str, Any],
    cache_dir: str | Path,
    extract_func,
    enable_cache: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache_path, metadata, cache_hash = sample_flow_cache_path(sample, settings, cache_dir)
    cache_info = {
        "sample_id": str(sample.get("sample_id")),
        "cache_key": cache_hash,
        "cache_path": str(cache_path),
        "cache_status": "disabled" if not enable_cache else "miss",
        "n_rows": 0,
    }
    if enable_cache and cache_path.exists():
        try:
            raw_flow_df, _ = load_sample_flow_cache(cache_path)
            raw_flow_df = apply_sample_metadata_to_raw_flow(raw_flow_df, sample)
            cache_info["cache_status"] = "hit" if len(raw_flow_df) else "hit_empty"
            cache_info["n_rows"] = int(len(raw_flow_df))
            return raw_flow_df, cache_info
        except Exception as exc:
            warnings.warn(f"flow cache load failed for sample_id={sample.get('sample_id')}: {exc}. Recomputing.")
            cache_info["cache_status"] = "load_failed"

    raw_flow_df = extract_func(pd.Series(sample))
    if raw_flow_df is None:
        raw_flow_df = pd.DataFrame()
    raw_flow_df = apply_sample_metadata_to_raw_flow(raw_flow_df, sample)
    cache_info["n_rows"] = int(len(raw_flow_df))
    if enable_cache:
        save_sample_flow_cache(cache_path, metadata, raw_flow_df)
        cache_info["cache_status"] = "saved" if cache_info["cache_status"] == "miss" else f"{cache_info['cache_status']}_saved"
    return raw_flow_df, cache_info


def save_sample_flow_cache_for_sample(
    sample: pd.Series | dict[str, Any],
    *,
    settings: dict[str, Any],
    cache_dir: str | Path,
    raw_flow_df: pd.DataFrame | None,
    enable_cache: bool = True,
) -> dict[str, Any]:
    cache_path, metadata, cache_hash = sample_flow_cache_path(sample, settings, cache_dir)
    cache_info = {
        "sample_id": str(sample.get("sample_id")),
        "cache_key": cache_hash,
        "cache_path": str(cache_path),
        "cache_status": "disabled" if not enable_cache else "saved",
        "n_rows": int(len(raw_flow_df)) if raw_flow_df is not None else 0,
    }
    if enable_cache:
        save_sample_flow_cache(cache_path, metadata, apply_sample_metadata_to_raw_flow(raw_flow_df if raw_flow_df is not None else pd.DataFrame(), sample))
    return cache_info


def load_sample_flow_cache_grid_parts(
    sample: pd.Series | dict[str, Any],
    *,
    settings: dict[str, Any],
    cache_dir: str | Path,
    views: tuple[str, ...] = ("front", "rear"),
    enable_cache: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    cache_path, _, cache_hash = sample_flow_cache_path(sample, settings, cache_dir)
    sample_id = str(sample.get("sample_id", "unknown"))
    cache_info = {
        "sample_id": sample_id,
        "cache_key": cache_hash,
        "cache_path": str(cache_path),
        "cache_status": "disabled" if not enable_cache else "miss",
    }
    if not enable_cache or not cache_path.exists():
        return {}, cache_info
    try:
        raw_flow_df, _ = load_sample_flow_cache(cache_path)
        parts = {
            view: raw_flow_to_visualization_grid(raw_flow_df, sample_id=sample_id, view=view)
            for view in views
        }
        parts = {view: part for view, part in parts.items() if len(part)}
        cache_info["cache_status"] = "hit" if parts else "hit_without_raw_flow"
        return parts, cache_info
    except Exception as exc:
        cache_info["cache_status"] = f"load_failed: {exc!r}"
        return {}, cache_info


def load_sample_feature_cache_grid_parts(
    sample: pd.Series | dict[str, Any],
    *,
    settings: dict[str, Any],
    cache_dir: str | Path,
    views: tuple[str, ...] = ("front", "rear"),
    enable_cache: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load shared sample_feature_cache raw flow as visualization grid parts."""
    cache_path, _, cache_hash = sample_feature_cache_path(sample, settings, cache_dir)
    sample_id = str(sample.get("sample_id", "unknown"))
    cache_info = {
        "sample_id": sample_id,
        "cache_key": cache_hash,
        "cache_path": str(cache_path),
        "cache_status": "disabled" if not enable_cache else "miss",
    }
    if not enable_cache or not cache_path.exists():
        return {}, cache_info
    try:
        _, raw_flow_df, _ = load_sample_feature_cache(cache_path)
        parts = {
            view: raw_flow_to_visualization_grid(raw_flow_df, sample_id=sample_id, view=view)
            for view in views
        }
        parts = {view: part for view, part in parts.items() if len(part)}
        cache_info["cache_status"] = "hit" if parts else "hit_without_raw_flow"
        return parts, cache_info
    except Exception as exc:
        cache_info["cache_status"] = f"load_failed: {exc!r}"
        return {}, cache_info


def _numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default).astype(float)


def _first_value(df: pd.DataFrame, column: str, default: Any = np.nan) -> Any:
    if column not in df.columns or df.empty:
        return default
    values = df[column].dropna()
    return values.iat[0] if len(values) else default


def visualization_grid_to_raw_flow(
    grid_df: pd.DataFrame,
    *,
    sample_id: str,
    view: str,
    flow_sample_sec: float,
) -> pd.DataFrame:
    """Convert long visualization grid-flow rows to the shared wide raw-flow format."""
    if grid_df is None or grid_df.empty:
        return pd.DataFrame(columns=["video_id", "time", "time_bin"])

    work = grid_df.copy()
    prefix = str(view).lower().strip()
    time_col = "time_sec" if "time_sec" in work.columns else "time"
    work["time"] = _numeric_series(work, time_col, 0.0)
    grid_row_values = work["grid_row"] if "grid_row" in work.columns else work["grid_y"] if "grid_y" in work.columns else work["gy"] if "gy" in work.columns else pd.Series(0, index=work.index)
    grid_col_values = work["grid_col"] if "grid_col" in work.columns else work["grid_x"] if "grid_x" in work.columns else work["gx"] if "gx" in work.columns else pd.Series(0, index=work.index)
    work["grid_row"] = pd.to_numeric(grid_row_values, errors="coerce").fillna(0).astype(int)
    work["grid_col"] = pd.to_numeric(grid_col_values, errors="coerce").fillna(0).astype(int)
    work["flow_x"] = _numeric_series(work, "flow_x" if "flow_x" in work.columns else "flow_dx_mean", 0.0)
    work["flow_y"] = _numeric_series(work, "flow_y" if "flow_y" in work.columns else "flow_dy_mean", 0.0)
    if "flow_mag_mean" in work.columns:
        work["flow_mag"] = _numeric_series(work, "flow_mag_mean", 0.0)
    elif "flow_magnitude_mean" in work.columns:
        work["flow_mag"] = _numeric_series(work, "flow_magnitude_mean", 0.0)
    else:
        work["flow_mag"] = np.hypot(work["flow_x"], work["flow_y"])
    work["valid_ratio"] = _numeric_series(work, "valid_ratio", 1.0).clip(0.0, 1.0)
    work["flow_reliable_ratio"] = _numeric_series(work, "flow_reliable_ratio", 0.0).clip(0.0, 1.0)
    if "flow_backward_consistency" in work.columns:
        work["flow_backward_consistency"] = _numeric_series(work, "flow_backward_consistency", 0.0).clip(0.0, 1.0)
    else:
        work["flow_backward_consistency"] = work["flow_reliable_ratio"]
    work["flow_edge_strength"] = _numeric_series(work, "flow_edge_strength", 0.0).clip(lower=0.0)
    work["flow_edge_density"] = _numeric_series(work, "flow_edge_density", 0.0).clip(0.0, 1.0)
    work["flow_edge_confidence"] = _numeric_series(work, "flow_edge_confidence", 0.0).clip(0.0, 1.0)
    work["flow_coherence_confidence"] = _numeric_series(work, "flow_coherence_confidence", 1.0).clip(0.0, 1.0)
    work["flow_measurement_confidence"] = _numeric_series(work, "flow_measurement_confidence", 0.0).clip(0.0, 1.0)
    work["flow_failed"] = 0.0

    width = max(float(flow_sample_sec), 1e-6)
    rows: list[dict[str, Any]] = []
    for time_sec, part in work.groupby("time", sort=True):
        row: dict[str, Any] = {
            "video_id": str(sample_id),
            "time": float(time_sec),
            "time_bin": int(round(float(time_sec) / width)),
        }
        x_values = part["flow_x"].to_numpy(dtype=float)
        y_values = part["flow_y"].to_numpy(dtype=float)
        mag_values = part["flow_mag"].to_numpy(dtype=float)
        angles = np.arctan2(y_values, x_values) if len(part) else np.asarray([], dtype=float)
        row[f"{prefix}_flow_mag_mean"] = float(np.mean(mag_values)) if mag_values.size else 0.0
        row[f"{prefix}_flow_mag_std"] = float(np.std(mag_values)) if mag_values.size else 0.0
        row[f"{prefix}_flow_mag_max"] = float(np.max(mag_values)) if mag_values.size else 0.0
        row[f"{prefix}_flow_angle_mean"] = float(np.mean(angles)) if angles.size else 0.0
        row[f"{prefix}_flow_angle_std"] = float(np.std(angles)) if angles.size else 0.0
        row[f"{prefix}_flow_x_mean"] = float(np.mean(x_values)) if x_values.size else 0.0
        row[f"{prefix}_flow_x_std"] = float(np.std(x_values)) if x_values.size else 0.0
        row[f"{prefix}_flow_y_mean"] = float(np.mean(y_values)) if y_values.size else 0.0
        row[f"{prefix}_flow_y_std"] = float(np.std(y_values)) if y_values.size else 0.0
        row[f"{prefix}_flow_failed"] = 0.0
        row[f"{prefix}_flow_reliable_ratio"] = float(part["flow_reliable_ratio"].mean()) if len(part) else 0.0
        row[f"{prefix}_flow_backward_consistency"] = float(part["flow_backward_consistency"].mean()) if len(part) else 0.0
        row[f"{prefix}_flow_edge_confidence"] = float(part["flow_edge_confidence"].mean()) if len(part) else 0.0
        row[f"{prefix}_flow_coherence_confidence"] = float(part["flow_coherence_confidence"].mean()) if len(part) else 1.0
        row[f"{prefix}_flow_measurement_confidence"] = float(part["flow_measurement_confidence"].mean()) if len(part) else 0.0
        if "flow_mode" in part.columns:
            row[f"{prefix}_flow_mode"] = str(_first_value(part, "flow_mode", "observed_update"))
        for source_col, target_suffix in [
            ("flow_confidence", "flow_confidence"),
            ("flow_observed_dt_sec", "flow_observed_dt_sec"),
            ("flow_source_start_sec", "flow_source_start_sec"),
            ("flow_source_end_sec", "flow_source_end_sec"),
            ("flow_gap_capture_frames", "flow_gap_capture_frames"),
            ("flow_interpolated", "flow_interpolated"),
            ("flow_hold", "flow_hold"),
            ("flow_update_frame_index", "flow_update_frame_index"),
            ("flow_edge_strength", "flow_edge_strength"),
            ("flow_edge_density", "flow_edge_density"),
        ]:
            if source_col in part.columns:
                row[f"{prefix}_{target_suffix}"] = float(_numeric_series(part, source_col, np.nan).mean()) if len(part) else np.nan
        if "source_dt" in part.columns:
            row[f"{prefix}_flow_source_dt"] = float(_numeric_series(part, "source_dt", 0.0).mean()) if len(part) else 0.0
        if "flow_window_frame_count" in part.columns:
            row[f"{prefix}_flow_window_frame_count"] = float(_numeric_series(part, "flow_window_frame_count", 0.0).mean()) if len(part) else 0.0
        for cell in part.itertuples(index=False):
            gy, gx = int(cell.grid_row), int(cell.grid_col)
            row[f"{prefix}_flow_cell_{gy}_{gx}_mag_mean"] = float(cell.flow_mag)
            row[f"{prefix}_flow_cell_{gy}_{gx}_mag_std"] = 0.0
            row[f"{prefix}_flow_cell_{gy}_{gx}_x_mean"] = float(cell.flow_x)
            row[f"{prefix}_flow_cell_{gy}_{gx}_y_mean"] = float(cell.flow_y)
            row[f"{prefix}_flow_cell_{gy}_{gx}_valid_ratio"] = float(cell.valid_ratio)
            row[f"{prefix}_flow_cell_{gy}_{gx}_flow_reliable_ratio"] = float(cell.flow_reliable_ratio)
            row[f"{prefix}_flow_cell_{gy}_{gx}_flow_backward_consistency"] = float(cell.flow_backward_consistency)
            row[f"{prefix}_flow_cell_{gy}_{gx}_flow_edge_strength"] = float(cell.flow_edge_strength)
            row[f"{prefix}_flow_cell_{gy}_{gx}_flow_edge_density"] = float(cell.flow_edge_density)
            row[f"{prefix}_flow_cell_{gy}_{gx}_flow_edge_confidence"] = float(cell.flow_edge_confidence)
            row[f"{prefix}_flow_cell_{gy}_{gx}_flow_coherence_confidence"] = float(cell.flow_coherence_confidence)
            row[f"{prefix}_flow_cell_{gy}_{gx}_flow_measurement_confidence"] = float(cell.flow_measurement_confidence)
            row[f"{prefix}_flow_cell_{gy}_{gx}_source_dt"] = float(getattr(cell, "source_dt", np.nan))
            row[f"{prefix}_flow_cell_{gy}_{gx}_flow_window_frame_count"] = float(getattr(cell, "flow_window_frame_count", np.nan))
            row[f"{prefix}_flow_cell_{gy}_{gx}_forward_backward_error_mean"] = float(getattr(cell, "flow_forward_backward_error_mean", np.nan))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["video_id", "time"]).reset_index(drop=True)


def merge_raw_flow_dfs(raw_flow_dfs: list[pd.DataFrame], *, time_bin_sec: float | None = None) -> pd.DataFrame:
    valid = [df.copy() for df in raw_flow_dfs if df is not None and len(df)]
    if not valid:
        return pd.DataFrame(columns=["video_id", "time", "time_bin"])
    merged = valid[0]
    for other in valid[1:]:
        merged = merged.merge(other.drop(columns=["time_bin"], errors="ignore"), on=["video_id", "time"], how="outer")
    merged = merged.sort_values(["video_id", "time"]).reset_index(drop=True)
    if time_bin_sec is not None and "time" in merged.columns:
        merged["time_bin"] = np.round(pd.to_numeric(merged["time"], errors="coerce").fillna(0.0) / max(float(time_bin_sec), 1e-6)).astype(int)
    return merged


def visualization_grid_parts_to_raw_flow(
    grid_parts: Mapping[str, pd.DataFrame],
    *,
    sample_id: str,
    flow_sample_sec: float,
) -> pd.DataFrame:
    raw_parts = [
        visualization_grid_to_raw_flow(part, sample_id=sample_id, view=view, flow_sample_sec=flow_sample_sec)
        for view, part in grid_parts.items()
        if part is not None and len(part)
    ]
    return merge_raw_flow_dfs(raw_parts, time_bin_sec=flow_sample_sec)


def reset_raw_flow_time_bin(raw_flow_df: pd.DataFrame, *, time_bin_sec: float) -> pd.DataFrame:
    work = raw_flow_df.copy()
    if "time" in work.columns:
        work["time_bin"] = np.round(pd.to_numeric(work["time"], errors="coerce").fillna(0.0) / max(float(time_bin_sec), 1e-6)).astype(int)
    elif "time_bin" not in work.columns:
        work["time_bin"] = 0
    return work


def raw_flow_to_visualization_grid(raw_flow_df: pd.DataFrame, *, sample_id: str, view: str) -> pd.DataFrame:
    """Convert shared wide raw_flow cache into the visualization notebook grid format."""
    if raw_flow_df is None or raw_flow_df.empty:
        return pd.DataFrame()

    def to_float(value: Any, default: float = 0.0) -> float:
        value = pd.to_numeric(value, errors="coerce")
        return float(value) if pd.notna(value) else float(default)

    prefix = str(view).lower().strip()
    rows: list[dict[str, Any]] = []
    for _, row in raw_flow_df.sort_values("time").iterrows():
        for col in raw_flow_df.columns:
            marker = f"{prefix}_flow_cell_"
            if not col.startswith(marker) or not col.endswith("_x_mean"):
                continue
            suffix = col[len(marker):-len("_x_mean")]
            try:
                gy_str, gx_str = suffix.split("_", 1)
                gy, gx = int(gy_str), int(gx_str)
            except ValueError:
                continue
            y_col = f"{prefix}_flow_cell_{gy}_{gx}_y_mean"
            mag_col = f"{prefix}_flow_cell_{gy}_{gx}_mag_mean"
            valid_col = f"{prefix}_flow_cell_{gy}_{gx}_valid_ratio"
            reliable_col = f"{prefix}_flow_cell_{gy}_{gx}_flow_reliable_ratio"
            consistency_col = f"{prefix}_flow_cell_{gy}_{gx}_flow_backward_consistency"
            edge_strength_col = f"{prefix}_flow_cell_{gy}_{gx}_flow_edge_strength"
            edge_density_col = f"{prefix}_flow_cell_{gy}_{gx}_flow_edge_density"
            edge_confidence_col = f"{prefix}_flow_cell_{gy}_{gx}_flow_edge_confidence"
            coherence_confidence_col = f"{prefix}_flow_cell_{gy}_{gx}_flow_coherence_confidence"
            measurement_confidence_col = f"{prefix}_flow_cell_{gy}_{gx}_flow_measurement_confidence"
            source_dt_col = f"{prefix}_flow_cell_{gy}_{gx}_source_dt"
            frame_count_col = f"{prefix}_flow_cell_{gy}_{gx}_flow_window_frame_count"
            error_col = f"{prefix}_flow_cell_{gy}_{gx}_forward_backward_error_mean"
            x = to_float(row.get(col, 0.0))
            y = to_float(row.get(y_col, 0.0))
            magnitude = to_float(row.get(mag_col, np.nan), default=np.nan)
            if not np.isfinite(magnitude):
                magnitude = float(np.hypot(x, y))
            valid_ratio = to_float(row.get(valid_col, 1.0), default=1.0)
            reliable_ratio = to_float(row.get(reliable_col, np.nan), default=np.nan)
            consistency = to_float(row.get(consistency_col, row.get(f"{prefix}_flow_backward_consistency", reliable_ratio)), default=0.0)
            edge_strength = to_float(row.get(edge_strength_col, row.get(f"{prefix}_flow_edge_strength", 0.0)), default=0.0)
            edge_density = to_float(row.get(edge_density_col, row.get(f"{prefix}_flow_edge_density", 0.0)), default=0.0)
            edge_confidence = to_float(row.get(edge_confidence_col, row.get(f"{prefix}_flow_edge_confidence", 0.0)), default=0.0)
            coherence_confidence = to_float(row.get(coherence_confidence_col, row.get(f"{prefix}_flow_coherence_confidence", 1.0)), default=1.0)
            measurement_confidence = to_float(row.get(measurement_confidence_col, row.get(f"{prefix}_flow_measurement_confidence", 0.0)), default=0.0)
            source_dt = to_float(row.get(source_dt_col, row.get(f"{prefix}_flow_source_dt", np.nan)), default=np.nan)
            frame_count = to_float(row.get(frame_count_col, row.get(f"{prefix}_flow_window_frame_count", np.nan)), default=np.nan)
            fb_error = to_float(row.get(error_col, np.nan), default=np.nan)
            flow_mode = row.get(f"{prefix}_flow_mode", np.nan)
            rows.append({
                "sample_id": str(sample_id),
                "view": prefix,
                "time": to_float(row.get("time", 0.0)),
                "grid_row": gy,
                "grid_col": gx,
                "grid_y": gy,
                "grid_x": gx,
                "flow_x": x,
                "flow_y": y,
                "flow_magnitude_mean": magnitude,
                "valid_ratio": valid_ratio,
                "flow_reliable_ratio": reliable_ratio,
                "flow_backward_consistency": consistency,
                "flow_edge_strength": edge_strength,
                "flow_edge_density": edge_density,
                "flow_edge_confidence": edge_confidence,
                "flow_coherence_confidence": coherence_confidence,
                "flow_measurement_confidence": measurement_confidence,
                "source_dt": source_dt,
                "flow_window_frame_count": frame_count,
                "flow_forward_backward_error_mean": fb_error,
                "flow_mode": str(flow_mode) if pd.notna(flow_mode) else "observed_update",
                "flow_confidence": to_float(row.get(f"{prefix}_flow_confidence", np.nan), default=np.nan),
                "flow_observed_dt_sec": to_float(row.get(f"{prefix}_flow_observed_dt_sec", np.nan), default=np.nan),
                "flow_source_start_sec": to_float(row.get(f"{prefix}_flow_source_start_sec", np.nan), default=np.nan),
                "flow_source_end_sec": to_float(row.get(f"{prefix}_flow_source_end_sec", np.nan), default=np.nan),
                "flow_gap_capture_frames": to_float(row.get(f"{prefix}_flow_gap_capture_frames", np.nan), default=np.nan),
                "flow_interpolated": to_float(row.get(f"{prefix}_flow_interpolated", np.nan), default=np.nan),
                "flow_hold": to_float(row.get(f"{prefix}_flow_hold", np.nan), default=np.nan),
                "flow_update_frame_index": to_float(row.get(f"{prefix}_flow_update_frame_index", np.nan), default=np.nan),
            })
    return pd.DataFrame(rows)
