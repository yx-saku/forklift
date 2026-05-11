from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SYNC_SCORE_CONFIG = {
    "audio_anomaly_column": "audio_anomaly_score",
    "motion_anomaly_column": "motion_anomaly_score",
    "audio_event_column": "audio_event_score",
    "motion_event_column": "motion_event_score",
    "max_lag_windows": 2,
}

AUDIO_EVENT_RAW_COLUMNS = (
    "audio_log_rms",
    "audio_log_peak",
    "audio_spectral_flux",
    "audio_log_rms_abs_diff",
    "audio_high_freq_energy",
    "audio_centroid",
    "audio_bandwidth",
)
MOTION_EVENT_FEATURE_CANDIDATES = (
    "t_flow_x_broad_vib_score",
    "t_flow_y_broad_vib_score",
    "t_accel_impact_x_score",
    "t_accel_impact_y_score",
)


def fit_score_calibration(scores: np.ndarray | pd.Series, quantiles: tuple[float, float] = (0.5, 0.995)) -> dict[str, float]:
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"lower": 0.0, "upper": 1.0}
    lower_q, upper_q = sorted([float(quantiles[0]), float(quantiles[1])])
    lower = float(np.quantile(values, np.clip(lower_q, 0.0, 1.0)))
    upper = float(np.quantile(values, np.clip(upper_q, 0.0, 1.0)))
    if not np.isfinite(upper) or upper <= lower:
        upper = lower + 1e-6
    return {"lower": lower, "upper": upper}


def apply_score_calibration(scores: np.ndarray | pd.Series, calibration: dict[str, float] | None) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    calibration = calibration or {"lower": 0.0, "upper": 1.0}
    lower = float(calibration.get("lower", 0.0))
    upper = float(calibration.get("upper", lower + 1.0))
    return np.clip((values - lower) / max(upper - lower, 1e-6), 0.0, 1.0)


def isolation_forest_raw_scores(model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    if hasattr(model, "score_samples"):
        return -np.asarray(model.score_samples(X), dtype=float)
    if hasattr(model, "decision_function"):
        return -np.asarray(model.decision_function(X), dtype=float)
    raise TypeError("model must expose score_samples or decision_function")


def score_isolation_forest_artifact(features_df: pd.DataFrame, model_artifact: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    feature_names = list(model_artifact.get("feature_names", []))
    if not feature_names:
        raise ValueError("audio score model artifact has no feature_names")
    X = features_df.reindex(columns=feature_names, fill_value=0.0).replace([np.inf, -np.inf], np.nan)
    pipeline = model_artifact.get("preprocess_pipeline")
    if pipeline is not None:
        X_model = pipeline.transform(X)
    else:
        X_model = X.fillna(0.0)
    raw_scores = isolation_forest_raw_scores(model_artifact["model"], X_model)
    scores = apply_score_calibration(raw_scores, model_artifact.get("score_calibration"))
    return raw_scores, scores


def _score_sort_columns(df: pd.DataFrame) -> list[str]:
    if "video_id" in df.columns and "time_bin" in df.columns:
        return ["video_id", "time_bin"]
    if "video_id" in df.columns and "time" in df.columns:
        return ["video_id", "time"]
    if "time_bin" in df.columns:
        return ["time_bin"]
    if "time" in df.columns:
        return ["time"]
    return []


def _numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(float(default), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(float(default)).astype(float)


def _clip01(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def _audio_mel_columns(df: pd.DataFrame) -> list[str]:
    return sorted([col for col in df.columns if col.startswith("audio_mel_")])


def add_audio_event_raw_features(audio_df: pd.DataFrame, *, eps: float = 1e-8) -> pd.DataFrame:
    """Add raw audio-event features used by the cross-modal sync score."""
    out = audio_df.copy()
    if out.empty:
        for column in AUDIO_EVENT_RAW_COLUMNS:
            if column not in out.columns:
                out[column] = pd.Series(dtype=float)
        return out

    out["audio_log_rms"] = np.log(np.maximum(_numeric_series(out, "audio_rms").to_numpy(dtype=float), float(eps)))
    out["audio_log_peak"] = np.log(np.maximum(_numeric_series(out, "audio_peak").to_numpy(dtype=float), float(eps)))

    sort_cols = _score_sort_columns(out)
    ordered = out.sort_values(sort_cols) if sort_cols else out.copy()
    group_keys = ["video_id"] if "video_id" in ordered.columns else None

    log_rms_diff = pd.Series(0.0, index=ordered.index, dtype=float)
    if group_keys is None:
        log_rms_diff.loc[ordered.index] = pd.to_numeric(ordered["audio_log_rms"], errors="coerce").diff().abs().fillna(0.0)
    else:
        log_rms_diff.loc[ordered.index] = ordered.groupby(group_keys, sort=False)["audio_log_rms"].diff().abs().fillna(0.0)
    out["audio_log_rms_abs_diff"] = log_rms_diff.reindex(out.index).fillna(0.0).astype(float)

    mel_cols = _audio_mel_columns(out)
    spectral_flux = pd.Series(0.0, index=ordered.index, dtype=float)
    if mel_cols:
        def group_flux(group: pd.DataFrame) -> pd.Series:
            values = group[mel_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
            if values.size == 0:
                return pd.Series(0.0, index=group.index, dtype=float)
            previous = np.vstack([values[:1], values[:-1]])
            deltas = np.maximum(values - previous, 0.0)
            deltas[0, :] = 0.0
            return pd.Series(np.sqrt(np.mean(deltas * deltas, axis=1)), index=group.index, dtype=float)

        if group_keys is None:
            spectral_flux.loc[ordered.index] = group_flux(ordered)
        else:
            for _, group in ordered.groupby(group_keys, sort=False):
                spectral_flux.loc[group.index] = group_flux(group)
    out["audio_spectral_flux"] = spectral_flux.reindex(out.index).fillna(0.0).astype(float)
    return out


def fit_audio_event_score_calibration(
    audio_df: pd.DataFrame,
    *,
    quantiles: tuple[float, float] = (0.50, 0.95),
) -> dict[str, Any]:
    raw = add_audio_event_raw_features(audio_df)
    return {
        "quantiles": tuple(float(v) for v in quantiles),
        "features": {column: fit_score_calibration(_numeric_series(raw, column), quantiles) for column in AUDIO_EVENT_RAW_COLUMNS},
    }


def add_audio_event_scores(audio_df: pd.DataFrame, calibration: dict[str, Any] | None) -> pd.DataFrame:
    """Add percentile-normalized audio event scores from activity/change/timbre cues."""
    out = add_audio_event_raw_features(audio_df)
    feature_calibration = (calibration or {}).get("features", {})
    normalized: dict[str, np.ndarray] = {}
    for column in AUDIO_EVENT_RAW_COLUMNS:
        normalized[column] = apply_score_calibration(_numeric_series(out, column), feature_calibration.get(column))

    activity = np.maximum(normalized["audio_log_rms"], normalized["audio_log_peak"])
    change = np.maximum(normalized["audio_spectral_flux"], normalized["audio_log_rms_abs_diff"])
    timbre = np.maximum.reduce([
        normalized["audio_high_freq_energy"],
        normalized["audio_centroid"],
        normalized["audio_bandwidth"],
    ])
    out["audio_activity_score"] = _clip01(activity)
    out["audio_change_score"] = _clip01(change)
    out["audio_timbre_score"] = _clip01(timbre)
    out["audio_event_score"] = np.maximum(
        np.sqrt(np.clip(out["audio_activity_score"].to_numpy(dtype=float) * out["audio_change_score"].to_numpy(dtype=float), 0.0, 1.0)),
        np.sqrt(np.clip(out["audio_activity_score"].to_numpy(dtype=float) * out["audio_timbre_score"].to_numpy(dtype=float), 0.0, 1.0)),
    )
    return out


def motion_event_raw_scores_from_tensor(
    X: np.ndarray,
    feature_names: list[str] | tuple[str, ...],
    *,
    x_feature: str = "t_flow_x_broad_vib_score",
    y_feature: str = "t_flow_y_broad_vib_score",
    event_features: list[str] | tuple[str, ...] | None = None,
) -> np.ndarray:
    """Return max event-like motion feature value per motion window."""
    values = np.asarray(X, dtype=float)
    if values.size == 0:
        return np.zeros((0,), dtype=float)
    names = list(feature_names)
    raw_parts = []
    reduce_axes = tuple(range(1, values.ndim - 1))
    if event_features is None:
        event_features = tuple(dict.fromkeys((x_feature, y_feature, *MOTION_EVENT_FEATURE_CANDIDATES)))
    for feature in event_features:
        if feature in names:
            channel = values[..., names.index(feature)]
            raw_parts.append(np.nanmax(channel, axis=reduce_axes) if reduce_axes else channel)
    if not raw_parts:
        return np.zeros((values.shape[0],), dtype=float)
    return np.maximum.reduce([np.nan_to_num(part, nan=0.0, posinf=0.0, neginf=0.0) for part in raw_parts]).astype(float)


def selected_motion_event_features(feature_names: list[str] | tuple[str, ...]) -> list[str]:
    names = list(feature_names)
    return [feature for feature in MOTION_EVENT_FEATURE_CANDIDATES if feature in names]


def fit_event_score_calibration(
    values: np.ndarray | pd.Series,
    *,
    quantiles: tuple[float, float] = (0.50, 0.95),
) -> dict[str, float]:
    return fit_score_calibration(values, quantiles)


def compute_cross_modal_support_score(
    scored_df: pd.DataFrame,
    *,
    audio_anomaly_col: str = "audio_anomaly_score",
    motion_anomaly_col: str = "motion_anomaly_score",
    audio_event_col: str = "audio_event_score",
    motion_event_col: str = "motion_event_score",
    max_lag_windows: int = 2,
) -> pd.Series:
    """Score cross-modal support: audio anomaly with motion events or motion anomaly with audio events."""
    required = [audio_anomaly_col, motion_anomaly_col, audio_event_col, motion_event_col]
    if scored_df.empty or any(column not in scored_df.columns for column in required):
        return pd.Series(0.0, index=scored_df.index, name="sync_score")

    sync = pd.Series(0.0, index=scored_df.index, name="sync_score")
    sort_cols = _score_sort_columns(scored_df)
    ordered_df = scored_df.sort_values(sort_cols) if sort_cols else scored_df.copy()
    window = max(1, int(max_lag_windows) * 2 + 1)
    group_iter = ordered_df.groupby("video_id", sort=False) if "video_id" in ordered_df.columns else [(None, ordered_df)]
    for _, group in group_iter:
        audio_anomaly = _clip01(_numeric_series(group, audio_anomaly_col))
        motion_anomaly = _clip01(_numeric_series(group, motion_anomaly_col))
        audio_event = pd.Series(_clip01(_numeric_series(group, audio_event_col)), index=group.index)
        motion_event = pd.Series(_clip01(_numeric_series(group, motion_event_col)), index=group.index)
        nearby_audio_event = audio_event.rolling(window=window, center=True, min_periods=1).max().to_numpy(dtype=float)
        nearby_motion_event = motion_event.rolling(window=window, center=True, min_periods=1).max().to_numpy(dtype=float)
        audio_supported_by_motion = np.sqrt(np.clip(audio_anomaly * nearby_motion_event, 0.0, 1.0))
        motion_supported_by_audio = np.sqrt(np.clip(motion_anomaly * nearby_audio_event, 0.0, 1.0))
        sync.loc[group.index] = np.maximum(audio_supported_by_motion, motion_supported_by_audio)
    return sync


def add_composite_scores(
    scored_df: pd.DataFrame,
    *,
    sync_score_config: dict[str, Any] | None = None,
    final_score_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    scored = scored_df.copy()
    sync_score_config = {**DEFAULT_SYNC_SCORE_CONFIG, **(sync_score_config or {})}
    final_score_weights = final_score_weights or {
        "audio_anomaly_score": 0.45,
        "motion_anomaly_score": 0.35,
        "sync_score": 0.20,
    }
    scored["sync_score"] = compute_cross_modal_support_score(
        scored,
        audio_anomaly_col=str(sync_score_config.get("audio_anomaly_column", "audio_anomaly_score")),
        motion_anomaly_col=str(sync_score_config.get("motion_anomaly_column", "motion_anomaly_score")),
        audio_event_col=str(sync_score_config.get("audio_event_column", "audio_event_score")),
        motion_event_col=str(sync_score_config.get("motion_event_column", "motion_event_score")),
        max_lag_windows=int(sync_score_config.get("max_lag_windows", 2)),
    )
    final = np.zeros(len(scored), dtype=float)
    weight_sum = 0.0
    for column, weight in final_score_weights.items():
        weight = float(weight)
        if weight <= 0.0:
            continue
        if column not in scored.columns:
            scored[column] = 0.0
        final += pd.to_numeric(scored[column], errors="coerce").fillna(0.0).to_numpy(dtype=float) * weight
        weight_sum += weight
    scored["final_anomaly_score"] = np.clip(final / max(weight_sum, 1e-6), 0.0, 1.0)
    scored["anomaly_score"] = scored["final_anomaly_score"]
    if "video_id" in scored.columns:
        scored["anomaly_score_smooth"] = scored.groupby("video_id")["anomaly_score"].transform(lambda s: s.rolling(window=5, center=True, min_periods=1).mean())
    else:
        scored["anomaly_score_smooth"] = scored["anomaly_score"].rolling(window=5, center=True, min_periods=1).mean()
    return scored


def topk_mean(s: pd.Series, k: int = 5) -> float:
    return float(s.nlargest(min(int(k), len(s))).mean()) if len(s) else 0.0
