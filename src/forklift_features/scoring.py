from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


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


def compute_temporal_sync_score(
    scored_df: pd.DataFrame,
    *,
    audio_col: str = "audio_anomaly_score",
    motion_col: str = "motion_anomaly_score",
    max_lag_windows: int = 2,
) -> pd.Series:
    if scored_df.empty or audio_col not in scored_df.columns or motion_col not in scored_df.columns:
        return pd.Series(0.0, index=scored_df.index, name="sync_score")
    sync = pd.Series(0.0, index=scored_df.index, name="sync_score")
    sort_cols = ["video_id", "time_bin"] if "time_bin" in scored_df.columns else ["video_id", "time"]
    window = max(1, int(max_lag_windows) * 2 + 1)
    for _, group in scored_df.sort_values(sort_cols).groupby("video_id", sort=False):
        audio = pd.to_numeric(group[audio_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        motion = pd.to_numeric(group[motion_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        nearby_audio = audio.rolling(window=window, center=True, min_periods=1).max()
        nearby_motion = motion.rolling(window=window, center=True, min_periods=1).max()
        aligned = np.maximum(audio.to_numpy() * nearby_motion.to_numpy(), motion.to_numpy() * nearby_audio.to_numpy())
        sync.loc[group.index] = np.sqrt(np.clip(aligned, 0.0, 1.0))
    return sync


def add_composite_scores(
    scored_df: pd.DataFrame,
    *,
    sync_score_config: dict[str, Any] | None = None,
    final_score_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    scored = scored_df.copy()
    sync_score_config = sync_score_config or {
        "audio_score_column": "audio_anomaly_score",
        "motion_score_column": "motion_anomaly_score",
        "max_lag_windows": 2,
    }
    final_score_weights = final_score_weights or {
        "audio_anomaly_score": 0.45,
        "motion_anomaly_score": 0.35,
        "sync_score": 0.20,
    }
    scored["sync_score"] = compute_temporal_sync_score(
        scored,
        audio_col=str(sync_score_config.get("audio_score_column", "audio_anomaly_score")),
        motion_col=str(sync_score_config.get("motion_score_column", "motion_anomaly_score")),
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
