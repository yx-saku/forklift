from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_CONTENT_UPDATE_DIFF_THRESHOLD = 1.0
DEFAULT_CONTENT_UPDATE_RESIZE_WIDTH = 480
DEFAULT_CONTENT_UPDATE_GAUSSIAN_BLUR_KERNEL: int | None = None
DEFAULT_CONTENT_UPDATE_TILE_GRID_ROWS = 8
DEFAULT_CONTENT_UPDATE_TILE_GRID_COLS = 12
DEFAULT_CONTENT_UPDATE_TILE_TOP_FRACTION = 0.10
DEFAULT_CONTENT_UPDATE_TILE_ACTIVE_THRESHOLD_RATIO = 0.50
DEFAULT_CONTENT_UPDATE_TILE_MIN_ACTIVE_TILES = 1
DEFAULT_CONTENT_UPDATE_TILE_MIN_ACTIVE_RATIO = 0.01
DEFAULT_CONTENT_UPDATE_EDGE_WEIGHT = 0.50
DEFAULT_STATIC_SCENE_MIN_SEC = 0.5
DEFAULT_FLOW_TARGET_DT = 0.1
DEFAULT_FLOW_NORMALIZE_BY_DT = True


def resize_keep_aspect(frame: np.ndarray, width: int | None, *, allow_upscale: bool = False) -> np.ndarray:
    """Resize a frame to a target width while preserving aspect ratio."""
    if width is None or int(width) <= 0:
        return frame
    h, w = frame.shape[:2]
    target_width = int(width)
    if w == target_width or (w < target_width and not allow_upscale):
        return frame
    height = max(1, int(round(h * (target_width / max(w, 1)))))
    return cv2.resize(frame, (target_width, height), interpolation=cv2.INTER_AREA)


def prepare_gray_frame(
    frame_bgr: np.ndarray,
    *,
    resize_width: int | None = DEFAULT_CONTENT_UPDATE_RESIZE_WIDTH,
    gaussian_blur_kernel: int | None = DEFAULT_CONTENT_UPDATE_GAUSSIAN_BLUR_KERNEL,
) -> np.ndarray:
    """Convert a BGR frame into the lightweight grayscale image used for update detection."""
    frame_bgr = resize_keep_aspect(frame_bgr, resize_width)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if gaussian_blur_kernel is not None and int(gaussian_blur_kernel) >= 3:
        kernel = int(gaussian_blur_kernel)
        if kernel % 2 == 0:
            kernel += 1
        gray = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    return gray


def frame_mean_abs_diff(current_gray: np.ndarray, previous_gray: np.ndarray) -> float:
    """Return the mean absolute pixel difference between two grayscale frames."""
    if current_gray.shape != previous_gray.shape:
        current_gray = cv2.resize(
            current_gray,
            (previous_gray.shape[1], previous_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    diff = current_gray.astype(np.float32) - previous_gray.astype(np.float32)
    return float(np.mean(np.abs(diff))) if diff.size else float("inf")


def frame_tile_abs_diff_stats(
    current_gray: np.ndarray,
    previous_gray: np.ndarray,
    *,
    diff_threshold: float = DEFAULT_CONTENT_UPDATE_DIFF_THRESHOLD,
    grid_rows: int = DEFAULT_CONTENT_UPDATE_TILE_GRID_ROWS,
    grid_cols: int = DEFAULT_CONTENT_UPDATE_TILE_GRID_COLS,
    top_fraction: float = DEFAULT_CONTENT_UPDATE_TILE_TOP_FRACTION,
    active_threshold_ratio: float = DEFAULT_CONTENT_UPDATE_TILE_ACTIVE_THRESHOLD_RATIO,
    min_active_tiles: int = DEFAULT_CONTENT_UPDATE_TILE_MIN_ACTIVE_TILES,
    min_active_ratio: float = DEFAULT_CONTENT_UPDATE_TILE_MIN_ACTIVE_RATIO,
) -> dict[str, float]:
    """Return tile-based frame difference stats for local content-update detection."""
    if current_gray.shape != previous_gray.shape:
        current_gray = cv2.resize(
            current_gray,
            (previous_gray.shape[1], previous_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    diff = np.abs(current_gray.astype(np.float32) - previous_gray.astype(np.float32))
    if diff.size == 0:
        return {
            "score": float("inf"),
            "tile_top_mean": float("inf"),
            "tile_p95": float("inf"),
            "tile_max": float("inf"),
            "tile_active_count": 0.0,
            "tile_active_ratio": 0.0,
        }

    rows = max(1, int(grid_rows))
    cols = max(1, int(grid_cols))
    row_edges = np.linspace(0, diff.shape[0], rows + 1, dtype=int)
    col_edges = np.linspace(0, diff.shape[1], cols + 1, dtype=int)
    tile_scores: list[float] = []
    for row_index in range(rows):
        row_start = int(row_edges[row_index])
        row_end = int(row_edges[row_index + 1])
        if row_end <= row_start:
            continue
        for col_index in range(cols):
            col_start = int(col_edges[col_index])
            col_end = int(col_edges[col_index + 1])
            if col_end <= col_start:
                continue
            tile = diff[row_start:row_end, col_start:col_end]
            if tile.size:
                tile_scores.append(float(np.mean(tile)))

    scores = np.asarray(tile_scores, dtype=float)
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size == 0:
        return {
            "score": np.nan,
            "tile_top_mean": np.nan,
            "tile_p95": np.nan,
            "tile_max": np.nan,
            "tile_active_count": 0.0,
            "tile_active_ratio": 0.0,
        }

    top_count = max(1, int(np.ceil(finite_scores.size * float(np.clip(top_fraction, 0.0, 1.0)))))
    sorted_scores = np.sort(finite_scores)
    tile_top_mean = float(np.mean(sorted_scores[-top_count:]))
    tile_p95 = float(np.percentile(finite_scores, 95))
    tile_max = float(np.max(finite_scores))
    active_threshold = max(float(diff_threshold) * max(float(active_threshold_ratio), 0.0), 1e-9)
    active_count = int(np.count_nonzero(finite_scores >= active_threshold))
    active_ratio = float(active_count / finite_scores.size) if finite_scores.size else 0.0
    active_enough = active_count >= max(1, int(min_active_tiles)) and active_ratio >= max(0.0, float(min_active_ratio))
    score = max(tile_top_mean, tile_p95, tile_max) if active_enough else 0.0
    return {
        "score": float(score),
        "tile_top_mean": tile_top_mean,
        "tile_p95": tile_p95,
        "tile_max": tile_max,
        "tile_active_count": float(active_count),
        "tile_active_ratio": active_ratio,
    }


def _tile_diff_summary(
    diff: np.ndarray,
    *,
    diff_threshold: float = DEFAULT_CONTENT_UPDATE_DIFF_THRESHOLD,
    grid_rows: int = DEFAULT_CONTENT_UPDATE_TILE_GRID_ROWS,
    grid_cols: int = DEFAULT_CONTENT_UPDATE_TILE_GRID_COLS,
    top_fraction: float = DEFAULT_CONTENT_UPDATE_TILE_TOP_FRACTION,
    active_threshold_ratio: float = DEFAULT_CONTENT_UPDATE_TILE_ACTIVE_THRESHOLD_RATIO,
) -> dict[str, float]:
    """Summarize a per-pixel difference image with local-excess tile stats."""
    if diff.size == 0:
        return {
            "median": np.nan,
            "top_mean": np.nan,
            "p95": np.nan,
            "max": np.nan,
            "local_excess": np.nan,
            "active_count": 0.0,
            "active_ratio": 0.0,
        }

    rows = max(1, int(grid_rows))
    cols = max(1, int(grid_cols))
    row_edges = np.linspace(0, diff.shape[0], rows + 1, dtype=int)
    col_edges = np.linspace(0, diff.shape[1], cols + 1, dtype=int)
    tile_scores: list[float] = []
    for row_index in range(rows):
        row_start = int(row_edges[row_index])
        row_end = int(row_edges[row_index + 1])
        if row_end <= row_start:
            continue
        for col_index in range(cols):
            col_start = int(col_edges[col_index])
            col_end = int(col_edges[col_index + 1])
            if col_end <= col_start:
                continue
            tile = diff[row_start:row_end, col_start:col_end]
            if tile.size:
                tile_scores.append(float(np.mean(tile)))

    scores = np.asarray(tile_scores, dtype=float)
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size == 0:
        return {
            "median": np.nan,
            "top_mean": np.nan,
            "p95": np.nan,
            "max": np.nan,
            "local_excess": np.nan,
            "active_count": 0.0,
            "active_ratio": 0.0,
        }

    top_count = max(1, int(np.ceil(finite_scores.size * float(np.clip(top_fraction, 0.0, 1.0)))))
    sorted_scores = np.sort(finite_scores)
    median = float(np.median(finite_scores))
    top_mean = float(np.mean(sorted_scores[-top_count:]))
    p95 = float(np.percentile(finite_scores, 95))
    maximum = float(np.max(finite_scores))
    active_threshold = max(float(diff_threshold) * max(float(active_threshold_ratio), 0.0), 1e-9)
    active_count = int(np.count_nonzero(np.maximum(finite_scores - median, 0.0) >= active_threshold))
    active_ratio = float(active_count / finite_scores.size) if finite_scores.size else 0.0
    local_excess = max(0.0, top_mean - median, p95 - median, maximum - median)
    return {
        "median": median,
        "top_mean": top_mean,
        "p95": p95,
        "max": maximum,
        "local_excess": float(local_excess),
        "active_count": float(active_count),
        "active_ratio": active_ratio,
    }


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def frame_local_update_stats(
    current_gray: np.ndarray,
    previous_gray: np.ndarray,
    *,
    diff_threshold: float = DEFAULT_CONTENT_UPDATE_DIFF_THRESHOLD,
    grid_rows: int = DEFAULT_CONTENT_UPDATE_TILE_GRID_ROWS,
    grid_cols: int = DEFAULT_CONTENT_UPDATE_TILE_GRID_COLS,
    top_fraction: float = DEFAULT_CONTENT_UPDATE_TILE_TOP_FRACTION,
    active_threshold_ratio: float = DEFAULT_CONTENT_UPDATE_TILE_ACTIVE_THRESHOLD_RATIO,
    edge_weight: float = DEFAULT_CONTENT_UPDATE_EDGE_WEIGHT,
) -> dict[str, float]:
    """Return local-excess stats that suppress global tone/compression drift.

    The main score is based on how much the strongest tiles exceed the median
    tile change. Uniform whole-frame tone shifts therefore stay low, while a
    small moving region can still score high.
    """
    if current_gray.shape != previous_gray.shape:
        current_gray = cv2.resize(
            current_gray,
            (previous_gray.shape[1], previous_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    gray_diff = np.abs(current_gray.astype(np.float32) - previous_gray.astype(np.float32))
    gray_summary = _tile_diff_summary(
        gray_diff,
        diff_threshold=diff_threshold,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        top_fraction=top_fraction,
        active_threshold_ratio=active_threshold_ratio,
    )

    local_excess = float(gray_summary["local_excess"])
    try:
        edge_weight_value = float(edge_weight)
    except (TypeError, ValueError):
        edge_weight_value = 0.0
    if not np.isfinite(edge_weight_value):
        edge_weight_value = 0.0

    edge_local_excess = 0.0
    edge_active_count = 0.0
    edge_active_ratio = 0.0
    edge_global_background_diff = np.nan
    if edge_weight_value > 0.0:
        edge_diff = np.abs(_gradient_magnitude(current_gray) - _gradient_magnitude(previous_gray))
        edge_summary = _tile_diff_summary(
            edge_diff,
            diff_threshold=diff_threshold,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            top_fraction=top_fraction,
            active_threshold_ratio=active_threshold_ratio,
        )
        edge_local_excess = float(edge_summary["local_excess"])
        edge_active_count = float(edge_summary["active_count"])
        edge_active_ratio = float(edge_summary["active_ratio"])
        edge_global_background_diff = float(edge_summary["median"])

    combined_score = max(local_excess, edge_weight_value * edge_local_excess)
    return {
        "score": float(combined_score),
        "local_excess": local_excess,
        "edge_local_excess": edge_local_excess,
        "global_background_diff": float(gray_summary["median"]),
        "tile_top_mean": float(gray_summary["top_mean"]),
        "tile_p95": float(gray_summary["p95"]),
        "tile_max": float(gray_summary["max"]),
        "active_tile_count": float(max(float(gray_summary["active_count"]), edge_active_count)),
        "active_tile_ratio": float(max(float(gray_summary["active_ratio"]), edge_active_ratio)),
        "edge_global_background_diff": edge_global_background_diff,
    }


def is_local_content_update(
    stats: dict[str, float],
    *,
    diff_threshold: float = DEFAULT_CONTENT_UPDATE_DIFF_THRESHOLD,
    min_active_tiles: int = DEFAULT_CONTENT_UPDATE_TILE_MIN_ACTIVE_TILES,
    min_active_ratio: float = DEFAULT_CONTENT_UPDATE_TILE_MIN_ACTIVE_RATIO,
) -> bool:
    """Return True when local structural change is strong enough to keep."""
    score = float(stats.get("score", np.nan))
    active_count = float(stats.get("active_tile_count", 0.0))
    active_ratio = float(stats.get("active_tile_ratio", 0.0))
    return bool(
        np.isfinite(score)
        and score >= float(diff_threshold)
        and active_count >= max(1, int(min_active_tiles))
        and active_ratio >= max(0.0, float(min_active_ratio))
    )


def is_content_update(diff_score: float, *, diff_threshold: float = DEFAULT_CONTENT_UPDATE_DIFF_THRESHOLD) -> bool:
    """Return True when a frame differs enough from the previous frame to keep."""
    return bool(np.isfinite(diff_score) and float(diff_score) >= float(diff_threshold))


def _select_update_peak_indices(
    score_array: np.ndarray,
    *,
    threshold: float,
    min_distance_frames: int,
) -> list[int]:
    candidate_indices: list[int] = []
    for index, score in enumerate(score_array):
        if not np.isfinite(score) or score < threshold:
            continue
        left = score_array[index - 1] if index > 0 else -np.inf
        right = score_array[index + 1] if index + 1 < score_array.size else -np.inf
        left = float(left) if np.isfinite(left) else -np.inf
        right = float(right) if np.isfinite(right) else -np.inf
        if float(score) >= left and float(score) >= right:
            candidate_indices.append(int(index))

    min_distance = max(1, int(min_distance_frames))
    selected_indices: list[int] = []
    for index in sorted(candidate_indices, key=lambda idx: (-float(score_array[idx]), int(idx))):
        if all(abs(int(index) - int(selected)) >= min_distance for selected in selected_indices):
            selected_indices.append(int(index))
    selected_indices.sort()
    return selected_indices


def find_update_peaks(
    scores: list[float] | np.ndarray,
    *,
    peak_percentile: float = 50.0,
    min_distance_frames: int = 2,
    min_score: float | None = None,
) -> dict[str, Any]:
    """Find local update peaks at or above the configured score threshold."""
    score_array = np.asarray(scores, dtype=float)
    finite_scores = score_array[np.isfinite(score_array)]
    if score_array.size == 0 or finite_scores.size == 0:
        return {
            "peak_indices": [],
            "peak_score_threshold": np.nan,
            "peak_score_percentile": np.nan,
            "peak_interval_count": 0,
            "peak_threshold_source": "empty_scores",
            "peak_threshold_attempts": 0,
        }

    peak_score_percentile = float(np.clip(peak_percentile, 0.0, 100.0))
    peak_threshold = float(np.percentile(finite_scores, peak_score_percentile))
    if min_score is not None and np.isfinite(min_score):
        peak_threshold = max(peak_threshold, float(min_score))
    all_peak_indices = _select_update_peak_indices(
        score_array,
        threshold=peak_threshold,
        min_distance_frames=min_distance_frames,
    )

    return {
        "peak_indices": all_peak_indices,
        "peak_score_threshold": peak_threshold,
        "peak_score_percentile": peak_score_percentile,
        "peak_interval_count": max(0, len(all_peak_indices) - 1),
        "peak_threshold_source": "fixed_percentile",
        "peak_threshold_attempts": 1,
    }


def estimate_fps_from_peak_intervals(
    peak_indices: list[int] | np.ndarray,
    *,
    input_fps: float,
    fallback_interval_sec: float = DEFAULT_FLOW_TARGET_DT,
    min_fps: int = 1,
    max_fps: int | None = None,
    min_peak_count: int = 5,
    continuity_gap_multiplier: float = 2.0,
    scores: list[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Estimate an integer update FPS from the longest continuous peak run."""
    source_fps = float(input_fps) if np.isfinite(input_fps) and float(input_fps) > 0.0 else float(min_fps)
    min_fps_int = max(1, int(min_fps))
    max_fps_int = int(max_fps) if max_fps is not None and int(max_fps) > 0 else int(round(source_fps))
    max_fps_int = max(min_fps_int, max_fps_int)

    peaks = np.asarray(sorted({int(index) for index in np.asarray(peak_indices, dtype=int)}), dtype=int)
    min_peak_count_int = max(2, int(min_peak_count))
    gap_multiplier = float(continuity_gap_multiplier) if np.isfinite(continuity_gap_multiplier) else 2.0
    gap_multiplier = gap_multiplier if gap_multiplier > 0.0 else 2.0
    peak_intervals = np.diff(peaks) if peaks.size >= 2 else np.asarray([], dtype=int)
    positive_peak_intervals = peak_intervals[peak_intervals > 0]
    min_gap_frames = float(np.min(positive_peak_intervals)) if positive_peak_intervals.size else np.nan
    # Allow repeated doubled gaps, but do not let a single high outlier bridge two runs.
    supported_gap_limit_frames = (
        float(np.percentile(positive_peak_intervals.astype(float), 75))
        if positive_peak_intervals.size
        else np.nan
    )
    gap_limit_frames = (
        float(min(min_gap_frames * gap_multiplier, supported_gap_limit_frames))
        if np.isfinite(min_gap_frames) and np.isfinite(supported_gap_limit_frames)
        else np.nan
    )

    selected_peak_indices: list[int] = []
    seed_peak_index = -1
    window_score = np.nan
    window_interval_std = np.nan
    interval_source = "fallback_peak_count"
    score_array = np.asarray(scores, dtype=float) if scores is not None else np.asarray([], dtype=float)

    def peak_score(peak_index: int) -> float:
        if 0 <= int(peak_index) < score_array.size and np.isfinite(score_array[int(peak_index)]):
            return float(max(score_array[int(peak_index)], 0.0))
        return 0.0

    if peaks.size >= min_peak_count_int and np.isfinite(gap_limit_frames):
        run_bounds: list[tuple[int, int]] = []
        left = 0
        for position in range(1, int(peaks.size)):
            if int(peaks[position] - peaks[position - 1]) > gap_limit_frames:
                run_bounds.append((int(left), int(position - 1)))
                left = int(position)
        run_bounds.append((int(left), int(peaks.size - 1)))

        best_run_key: tuple[int, int] | None = None
        for left, right in run_bounds:
            candidate_peaks = [int(index) for index in peaks[left : right + 1]]
            if len(candidate_peaks) < min_peak_count_int:
                continue
            candidate_intervals = np.diff(np.asarray(candidate_peaks, dtype=int))
            peak_scores = [peak_score(index) for index in candidate_peaks]
            candidate_score = float(np.sum(peak_scores))
            candidate_key = (int(len(candidate_peaks)), -int(left))
            if best_run_key is None or candidate_key > best_run_key:
                best_run_key = candidate_key
                selected_peak_indices = candidate_peaks
                window_score = candidate_score
                window_interval_std = float(np.std(candidate_intervals.astype(float))) if candidate_intervals.size else np.nan
                interval_source = "peak_continuity_run"
        if selected_peak_indices:
            seed_peak_index = int(selected_peak_indices[0])
        if not selected_peak_indices:
            interval_source = "fallback_insufficient_continuous_peak_run"
    elif peaks.size > 1:
        interval_source = "fallback_insufficient_peak_count"

    estimation_peak_indices = selected_peak_indices
    estimation_intervals = np.diff(np.asarray(estimation_peak_indices, dtype=int)) if len(estimation_peak_indices) >= 2 else np.asarray([], dtype=int)
    estimation_interval_start_indices = [int(index) for index in estimation_peak_indices[:-1]]
    estimation_interval_end_indices = [int(index) for index in estimation_peak_indices[1:]]
    interval_mean = float(np.mean(estimation_intervals.astype(float))) if estimation_intervals.size else np.nan
    interval_p50 = float(np.median(estimation_intervals)) if estimation_intervals.size else np.nan

    if np.isfinite(interval_mean) and interval_mean > 0.0:
        base_interval_for_fps = float(interval_mean)
        base_interval_frames = max(1, int(round(base_interval_for_fps)))
    else:
        fallback_interval = source_fps * float(fallback_interval_sec) if np.isfinite(fallback_interval_sec) else source_fps
        base_interval_for_fps = float(max(fallback_interval, 1e-12))
        base_interval_frames = max(1, int(round(base_interval_for_fps)))

    raw_estimated_fps = source_fps / float(base_interval_for_fps)
    estimated_fps = int(round(raw_estimated_fps)) if np.isfinite(raw_estimated_fps) else min_fps_int
    estimated_fps = max(min_fps_int, min(max_fps_int, estimated_fps))
    estimated_interval_frames = max(1, int(round(source_fps / float(estimated_fps))))
    return {
        "estimated_fps": int(estimated_fps),
        "estimated_interval_frames": int(estimated_interval_frames),
        "estimated_fps_source": interval_source,
        "raw_estimated_fps": float(raw_estimated_fps) if np.isfinite(raw_estimated_fps) else np.nan,
        "peak_interval_mean_frames": interval_mean,
        "peak_interval_mean_sec": float(interval_mean / source_fps) if np.isfinite(interval_mean) and source_fps > 0.0 else np.nan,
        "peak_interval_p50_frames": interval_p50,
        "peak_interval_p50_sec": float(interval_p50 / source_fps) if np.isfinite(interval_p50) and source_fps > 0.0 else np.nan,
        "fps_estimation_peak_indices": estimation_peak_indices,
        "fps_estimation_interval_start_indices": estimation_interval_start_indices,
        "fps_estimation_interval_end_indices": estimation_interval_end_indices,
        "fps_estimation_interval_frames": [int(value) for value in estimation_intervals],
        "fps_estimation_interval_count": int(estimation_intervals.size),
        "fps_estimation_run_peak_count": int(len(estimation_peak_indices)),
        "fps_estimation_min_peak_count": int(min_peak_count_int),
        "fps_estimation_seed_peak_index": int(seed_peak_index),
        "fps_estimation_min_gap_frames": float(min_gap_frames) if np.isfinite(min_gap_frames) else np.nan,
        "fps_estimation_gap_limit_frames": float(gap_limit_frames) if np.isfinite(gap_limit_frames) else np.nan,
        "fps_estimation_continuity_gap_multiplier": float(gap_multiplier),
        "fps_estimation_window_score": float(window_score) if np.isfinite(window_score) else np.nan,
        "fps_estimation_window_interval_std": float(window_interval_std) if np.isfinite(window_interval_std) else np.nan,
        "base_interval_raw_frames": float(base_interval_for_fps),
        "base_interval_frames": int(base_interval_frames),
        "base_interval_sec": float(base_interval_for_fps / source_fps) if source_fps > 0.0 else np.nan,
    }


def select_phase_by_update_score(
    scores: list[float] | np.ndarray,
    interval_frames: int,
    *,
    peak_indices: list[int] | np.ndarray | None = None,
    include_first: bool = True,
    include_last: bool = False,
) -> dict[str, Any]:
    """Select the interval phase whose sampled frames have the largest update score sum."""
    score_array = np.asarray(scores, dtype=float)
    if score_array.size == 0:
        return {
            "selected_indices": [],
            "phase_indices": [],
            "selected_phase": 0,
            "selected_phase_score": 0.0,
            "selected_phase_peak_count": 0,
        }

    interval = max(1, int(interval_frames))
    safe_scores = np.where(np.isfinite(score_array), np.maximum(score_array, 0.0), 0.0)
    peak_set = {int(index) for index in np.asarray(peak_indices if peak_indices is not None else [], dtype=int)}
    best_phase = 0
    best_indices: list[int] = []
    best_score = -np.inf
    best_peak_count = -1
    best_key: tuple[float, int, int] | None = None
    for phase in range(interval):
        candidate_indices = list(range(phase, int(score_array.size), interval))
        phase_score = float(np.sum(safe_scores[candidate_indices])) if candidate_indices else 0.0
        peak_count = int(sum(1 for index in candidate_indices if int(index) in peak_set))
        key = (phase_score, peak_count, -int(phase))
        if best_key is None or key > best_key:
            best_key = key
            best_phase = int(phase)
            best_indices = candidate_indices
            best_score = phase_score
            best_peak_count = peak_count

    selected_indices = set(best_indices)
    if include_first:
        selected_indices.add(0)
    if include_last:
        selected_indices.add(int(score_array.size) - 1)

    return {
        "selected_indices": sorted(int(index) for index in selected_indices),
        "phase_indices": [int(index) for index in best_indices],
        "selected_phase": int(best_phase),
        "selected_phase_score": float(best_score),
        "selected_phase_peak_count": int(best_peak_count),
    }


def select_phase_by_recomputed_update_score(
    gray_frames: list[np.ndarray] | tuple[np.ndarray, ...],
    interval_frames: int,
    *,
    diff_threshold: float = DEFAULT_CONTENT_UPDATE_DIFF_THRESHOLD,
    grid_rows: int = DEFAULT_CONTENT_UPDATE_TILE_GRID_ROWS,
    grid_cols: int = DEFAULT_CONTENT_UPDATE_TILE_GRID_COLS,
    top_fraction: float = DEFAULT_CONTENT_UPDATE_TILE_TOP_FRACTION,
    active_threshold_ratio: float = DEFAULT_CONTENT_UPDATE_TILE_ACTIVE_THRESHOLD_RATIO,
    edge_weight: float = DEFAULT_CONTENT_UPDATE_EDGE_WEIGHT,
    peak_indices: list[int] | np.ndarray | None = None,
    include_first: bool = True,
    include_last: bool = False,
) -> dict[str, Any]:
    """Select the phase by rescoring the frames that would actually be kept."""
    frame_count = int(len(gray_frames))
    if frame_count == 0:
        return {
            "selected_indices": [],
            "phase_indices": [],
            "selected_phase": 0,
            "selected_phase_score": 0.0,
            "selected_phase_peak_count": 0,
            "selected_phase_recomputed_pair_count": 0,
        }

    interval = max(1, int(interval_frames))
    peak_set = {int(index) for index in np.asarray(peak_indices if peak_indices is not None else [], dtype=int)}
    best_phase = 0
    best_phase_indices: list[int] = []
    best_selected_indices: list[int] = []
    best_score = -np.inf
    best_peak_count = -1
    best_pair_count = 0
    best_key: tuple[float, int, int, int] | None = None
    for phase in range(interval):
        phase_indices = list(range(int(phase), frame_count, interval))
        selected_indices = set(phase_indices)
        if include_first:
            selected_indices.add(0)
        if include_last:
            selected_indices.add(frame_count - 1)
        ordered_indices = sorted(index for index in selected_indices if 0 <= int(index) < frame_count)
        phase_score = 0.0
        pair_count = 0
        for previous_index, current_index in zip(ordered_indices[:-1], ordered_indices[1:]):
            stats = frame_local_update_stats(
                gray_frames[int(current_index)],
                gray_frames[int(previous_index)],
                diff_threshold=diff_threshold,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                top_fraction=top_fraction,
                active_threshold_ratio=active_threshold_ratio,
                edge_weight=edge_weight,
            )
            score = float(stats.get("score", np.nan))
            if np.isfinite(score):
                phase_score += max(score, 0.0)
            pair_count += 1
        peak_count = int(sum(1 for index in phase_indices if int(index) in peak_set))
        key = (float(phase_score), int(peak_count), int(pair_count), -int(phase))
        if best_key is None or key > best_key:
            best_key = key
            best_phase = int(phase)
            best_phase_indices = [int(index) for index in phase_indices]
            best_selected_indices = [int(index) for index in ordered_indices]
            best_score = float(phase_score)
            best_peak_count = int(peak_count)
            best_pair_count = int(pair_count)

    return {
        "selected_indices": best_selected_indices,
        "phase_indices": best_phase_indices,
        "selected_phase": int(best_phase),
        "selected_phase_score": float(best_score) if np.isfinite(best_score) else 0.0,
        "selected_phase_peak_count": int(best_peak_count),
        "selected_phase_recomputed_pair_count": int(best_pair_count),
    }


def content_update_output_fps(kept_count: int, duration_sec: float, fallback_fps: float, *, min_fps: float = 1e-6) -> float:
    """Compute playback FPS for a thinned video while keeping its original duration."""
    rate = float(kept_count) / float(duration_sec) if duration_sec > 0.0 and kept_count > 0 else np.nan
    output_fps = rate if np.isfinite(rate) and rate > 0.0 else float(fallback_fps)
    return max(float(output_fps), float(min_fps))


def summarize_content_update_view(
    prefix: str,
    *,
    enabled: bool,
    input_frame_pairs: int,
    kept_indices: list[int],
    duplicate_count: int,
    diff_scores: list[float],
    max_duplicate_run: int,
    max_time_since_content_update: float,
    source_duration_sec: float,
    fallback_fps: float,
    static_scene_min_sec: float = DEFAULT_STATIC_SCENE_MIN_SEC,
    min_output_fps: float = 1e-6,
) -> dict[str, Any]:
    """Build manifest fields for one front/rear content-update thinning result."""
    kept_count = int(len(kept_indices))
    scores = np.asarray(diff_scores, dtype=float)
    finite_scores = scores[np.isfinite(scores)]
    output_fps = content_update_output_fps(kept_count, source_duration_sec, fallback_fps, min_fps=min_output_fps)
    return {
        f"{prefix}_enabled": bool(enabled),
        f"{prefix}_input_frame_pairs": int(input_frame_pairs),
        f"{prefix}_kept_frame_pairs": kept_count,
        f"{prefix}_removed_frame_pairs": int(duplicate_count),
        f"{prefix}_removed_ratio": float(duplicate_count / input_frame_pairs) if input_frame_pairs else 0.0,
        f"{prefix}_kept_ratio": float(kept_count / input_frame_pairs) if input_frame_pairs else 0.0,
        f"{prefix}_output_fps": float(output_fps),
        f"{prefix}_diff_mean": float(np.mean(finite_scores)) if finite_scores.size else np.nan,
        f"{prefix}_diff_p95": float(np.percentile(finite_scores, 95)) if finite_scores.size else np.nan,
        f"{prefix}_max_duplicate_run": int(max_duplicate_run),
        f"{prefix}_max_time_since_content_update": float(max_time_since_content_update),
        f"{prefix}_static_scene_flag": int(max_time_since_content_update >= float(static_scene_min_sec)),
    }


def safe_video_fps(capture: cv2.VideoCapture, *, fallback: float = 30.0) -> float:
    """Read a valid FPS value from OpenCV, falling back when metadata is missing."""
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    return fps if np.isfinite(fps) and fps > 0.0 else float(fallback)


def write_selected_frames_mp4(
    input_video_path: str | Path,
    output_video_path: str | Path,
    frame_indices: list[int],
    *,
    output_fps: float,
    fourcc_name: str = "mp4v",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a new MP4 containing only selected source frame indices."""
    input_video_path = Path(input_video_path)
    output_video_path = Path(output_video_path)
    base = {
        "thinned_video_path": str(output_video_path),
        "thinned_video_fps": float(output_fps) if np.isfinite(output_fps) else np.nan,
        "thinned_video_frame_count": int(len(frame_indices)),
    }
    if not frame_indices:
        return {**base, "thinned_video_status": "no_content_frames"}
    if output_video_path.exists() and not overwrite:
        return {**base, "thinned_video_status": "exists"}

    capture = cv2.VideoCapture(str(input_video_path))
    if not capture.isOpened():
        return {**base, "thinned_video_status": "open_failed"}
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if frame_width <= 0 or frame_height <= 0:
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            return {**base, "thinned_video_status": "first_frame_read_failed"}
        frame_height, frame_width = frame.shape[:2]
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*str(fourcc_name))
    writer = cv2.VideoWriter(str(output_video_path), fourcc, float(output_fps), (frame_width, frame_height))
    if not writer.isOpened():
        capture.release()
        return {**base, "thinned_video_status": "writer_open_failed"}

    content_set = {int(idx) for idx in frame_indices}
    written_count = 0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in content_set:
                writer.write(frame)
                written_count += 1
            frame_index += 1
    finally:
        writer.release()
        capture.release()
    return {
        **base,
        "thinned_video_status": "saved" if written_count else "no_frames_written",
        "thinned_video_frame_count": int(written_count),
    }
