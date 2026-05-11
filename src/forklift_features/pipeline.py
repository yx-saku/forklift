from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import audio as feature_audio
from . import cache as feature_cache
from . import flow_extract, flow_tensor


def enabled_cameras(*, use_front: bool = True, use_rear: bool = True) -> list[str]:
    """Return the ordered camera list enabled for motion tensor extraction."""
    return [camera for camera, enabled in (("front", use_front), ("rear", use_rear)) if enabled]


def build_flow_cache_settings(
    *,
    cache_version: str,
    use_front: bool,
    use_rear: bool,
    flow_sample_sec: float,
    frame_resize_width: int | None,
    flow_analysis_scale: float,
    flow_grid: tuple[int, int],
    flow_reliable_error_threshold_px: float,
    flow_method: str,
) -> dict[str, Any]:
    """Build the raw-flow cache key shared by training, inference, and visualization."""
    return feature_cache.build_sample_flow_cache_settings(
        cache_version=cache_version,
        use_front=use_front,
        use_rear=use_rear,
        flow_sample_sec=flow_sample_sec,
        frame_resize_width=frame_resize_width,
        flow_analysis_scale=flow_analysis_scale,
        flow_grid=flow_grid,
        flow_reliable_error_threshold_px=flow_reliable_error_threshold_px,
        flow_method=flow_method,
    )


def extract_raw_flow_for_cache(
    sample: pd.Series | dict[str, Any],
    *,
    use_front: bool,
    use_rear: bool,
    flow_sample_sec: float,
    frame_resize_width: int | None,
    flow_analysis_scale: float,
    flow_grid: tuple[int, int],
    flow_reliable_error_threshold_px: float,
) -> pd.DataFrame:
    """Extract shared raw flow for a sample when the cache is missing."""
    return flow_extract.extract_sample_raw_flow(
        sample,
        use_front=use_front,
        use_rear=use_rear,
        flow_sample_sec=flow_sample_sec,
        frame_resize_width=frame_resize_width,
        flow_analysis_scale=flow_analysis_scale,
        flow_grid=flow_grid,
        reliable_error_threshold_px=flow_reliable_error_threshold_px,
    )


def load_or_extract_raw_flow(
    sample: pd.Series | dict[str, Any],
    *,
    flow_cache_settings: dict[str, Any],
    cache_dir: str | Path,
    use_front: bool,
    use_rear: bool,
    flow_sample_sec: float,
    frame_resize_width: int | None,
    flow_analysis_scale: float,
    flow_grid: tuple[int, int],
    flow_reliable_error_threshold_px: float,
    enable_cache: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load raw flow from cache or extract it with the current flow settings."""
    def extract_func(sample_series: pd.Series) -> pd.DataFrame:
        return extract_raw_flow_for_cache(
            sample_series,
            use_front=use_front,
            use_rear=use_rear,
            flow_sample_sec=flow_sample_sec,
            frame_resize_width=frame_resize_width,
            flow_analysis_scale=flow_analysis_scale,
            flow_grid=flow_grid,
            flow_reliable_error_threshold_px=flow_reliable_error_threshold_px,
        )

    raw_flow_df, cache_info = feature_cache.load_or_extract_sample_flow_cache(
        sample,
        settings=flow_cache_settings,
        cache_dir=cache_dir,
        extract_func=extract_func,
        enable_cache=enable_cache,
    )
    return feature_cache.reset_raw_flow_time_bin(raw_flow_df, time_bin_sec=flow_sample_sec), cache_info


def build_motion_windows(
    sample: pd.Series | dict[str, Any],
    *,
    flow_cache_settings: dict[str, Any],
    cache_dir: str | Path,
    use_front: bool,
    use_rear: bool,
    flow_sample_sec: float,
    frame_resize_width: int | None,
    flow_analysis_scale: float,
    flow_grid: tuple[int, int],
    flow_reliable_error_threshold_px: float,
    tensor_window_sec: float,
    tensor_hop_sec: float,
    motion_feature_names: Iterable[str],
    broad_vib_score_config: dict[str, Any] | None,
    reliable_soft_gate_config: dict[str, Any] | None = None,
    accel_impact_score_config: dict[str, Any] | None = None,
    label_default: str | None = None,
    enable_cache: bool = True,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build motion tensor windows and return the raw-flow cache metadata used."""
    sample_series = pd.Series(sample)
    raw_flow_df, cache_info = load_or_extract_raw_flow(
        sample_series,
        flow_cache_settings=flow_cache_settings,
        cache_dir=cache_dir,
        use_front=use_front,
        use_rear=use_rear,
        flow_sample_sec=flow_sample_sec,
        frame_resize_width=frame_resize_width,
        flow_analysis_scale=flow_analysis_scale,
        flow_grid=flow_grid,
        flow_reliable_error_threshold_px=flow_reliable_error_threshold_px,
        enable_cache=enable_cache,
    )
    X, meta = flow_tensor.build_flow_tensor_windows(
        raw_flow_df,
        cameras=enabled_cameras(use_front=use_front, use_rear=use_rear),
        flow_sample_sec=flow_sample_sec,
        window_sec=tensor_window_sec,
        hop_sec=tensor_hop_sec,
        grid=flow_grid,
        feature_names=motion_feature_names,
        broad_vib_score_config=broad_vib_score_config,
        reliable_soft_gate_config=reliable_soft_gate_config,
        accel_impact_score_config=accel_impact_score_config,
    )
    if len(meta):
        category = sample_series.get("category", label_default or "unknown")
        environment = sample_series.get("environment", "unknown")
        target_label = sample_series.get("plot_label", sample_series.get("sample_id", "unknown"))
        meta["category"] = category
        meta["environment"] = environment
        meta["target_category"] = category
        meta["target_environment"] = environment
        meta["target_label"] = target_label
        if label_default is not None:
            meta["label"] = label_default
    return X, meta, raw_flow_df, cache_info


def build_audio_features(
    sample: pd.Series | dict[str, Any],
    *,
    audio_sr: int,
    window_sec: float,
    hop_sec: float,
    n_mels: int,
    label_default: str | None = None,
) -> pd.DataFrame:
    """Extract audio features with consistent sample/category/environment metadata."""
    sample_series = pd.Series(sample)
    category = sample_series.get("category", label_default or "unknown")
    environment = sample_series.get("environment", "unknown")
    metadata: dict[str, Any] = {
        "category": category,
        "environment": environment,
        "target_category": category,
        "target_environment": environment,
        "target_label": sample_series.get("plot_label", sample_series.get("sample_id", "unknown")),
    }
    if label_default is not None:
        metadata["label"] = label_default
    return feature_audio.extract_audio_features(
        sample_series.get("audio_path"),
        sample_id=str(sample_series.get("sample_id", "unknown")),
        audio_sr=audio_sr,
        window_sec=window_sec,
        hop_sec=hop_sec,
        n_mels=n_mels,
        metadata=metadata,
    )


def normalize_score_time(df: pd.DataFrame, *, hop_sec: float) -> pd.DataFrame:
    """Normalize score tables to the common video_id/time/time_bin schema."""
    out = df.copy()
    if out.empty:
        return out
    if "video_id" not in out.columns and "sample_id" in out.columns:
        out["video_id"] = out["sample_id"].astype(str)
    if "sample_id" not in out.columns and "video_id" in out.columns:
        out["sample_id"] = out["video_id"].astype(str)
    if "time" not in out.columns and "window_center_sec" in out.columns:
        out["time"] = pd.to_numeric(out["window_center_sec"], errors="coerce").fillna(0.0)
    if "time_bin" not in out.columns:
        if "window_start_bin" in out.columns:
            out["time_bin"] = pd.to_numeric(out["window_start_bin"], errors="coerce").fillna(0).astype(int)
        else:
            time = pd.to_numeric(out["time"], errors="coerce").fillna(0.0)
            out["time_bin"] = np.round(time / max(float(hop_sec), 1e-6)).astype(int)
    return out


def combine_audio_motion_scores(audio_scores_df: pd.DataFrame, motion_scores_df: pd.DataFrame, *, hop_sec: float) -> pd.DataFrame:
    """Outer-join audio and motion window scores on video_id/time_bin."""
    audio_part = normalize_score_time(audio_scores_df, hop_sec=hop_sec)
    motion_part = normalize_score_time(motion_scores_df, hop_sec=hop_sec)
    key_cols = ["video_id", "time_bin"]
    if audio_part.empty:
        combined = motion_part.copy()
        combined["audio_anomaly_score"] = 0.0
        combined["audio_anomaly_score_raw"] = 0.0
        return combined
    if motion_part.empty:
        combined = audio_part.copy()
        combined["motion_anomaly_score"] = 0.0
        combined["motion_anomaly_score_raw"] = 0.0
        return combined

    meta_cols = ["sample_id", "target_category", "target_environment", "target_label", "category", "environment", "label", "time"]
    motion_meta_cols = [*meta_cols, "window_start_sec", "window_end_sec", "window_center_sec", "camera_count"]
    audio_keep = [c for c in audio_part.columns if c in key_cols or c.startswith("audio_") or c in meta_cols]
    motion_keep = [c for c in motion_part.columns if c in key_cols or c.startswith("motion_") or c in motion_meta_cols]
    combined = audio_part[audio_keep].merge(motion_part[motion_keep], on=key_cols, how="outer", suffixes=("_audio", "_motion"))
    for col in meta_cols:
        left, right = f"{col}_audio", f"{col}_motion"
        if left in combined.columns or right in combined.columns:
            combined[col] = combined[left] if left in combined.columns else np.nan
            if right in combined.columns:
                combined[col] = combined[col].combine_first(combined[right])
            combined = combined.drop(columns=[c for c in (left, right) if c in combined.columns])
    for col in ["audio_anomaly_score", "audio_anomaly_score_raw", "motion_anomaly_score", "motion_anomaly_score_raw"]:
        if col not in combined.columns:
            combined[col] = 0.0
        combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0.0).astype(float)
    fallback_time = combined["time_bin"] * float(hop_sec)
    combined["time"] = pd.to_numeric(combined.get("time", fallback_time), errors="coerce").fillna(fallback_time)
    return combined.sort_values(["video_id", "time_bin"]).reset_index(drop=True)
