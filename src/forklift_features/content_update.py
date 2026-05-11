from __future__ import annotations

from fractions import Fraction
from typing import Any

import cv2
import numpy as np


DEFAULT_CONTENT_UPDATE_RESIZE_WIDTH = 480
DEFAULT_CONTENT_UPDATE_GAUSSIAN_BLUR_KERNEL: int | None = None
DEFAULT_CONTENT_UPDATE_TILE_GRID_ROWS = 8
DEFAULT_CONTENT_UPDATE_TILE_GRID_COLS = 12
DEFAULT_CONTENT_UPDATE_TILE_TOP_FRACTION = 0.10
DEFAULT_FLOW_TARGET_DT = 0.1


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


def _tile_change_score(
    diff: np.ndarray,
    *,
    grid_rows: int = DEFAULT_CONTENT_UPDATE_TILE_GRID_ROWS,
    grid_cols: int = DEFAULT_CONTENT_UPDATE_TILE_GRID_COLS,
    top_fraction: float = DEFAULT_CONTENT_UPDATE_TILE_TOP_FRACTION,
) -> float:
    """Summarize a per-pixel difference image with a tile-based change score."""
    if diff.size == 0:
        return np.nan

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
        return np.nan

    top_count = max(1, int(np.ceil(finite_scores.size * float(np.clip(top_fraction, 0.0, 1.0)))))
    sorted_scores = np.sort(finite_scores)
    median = float(np.median(finite_scores))
    top_mean = float(np.mean(sorted_scores[-top_count:]))
    p95 = float(np.percentile(finite_scores, 95))
    maximum = float(np.max(finite_scores))
    return float(max(0.0, top_mean - median, p95 - median, maximum - median))


def frame_local_update_stats(
    current_gray: np.ndarray,
    previous_gray: np.ndarray,
    *,
    grid_rows: int = DEFAULT_CONTENT_UPDATE_TILE_GRID_ROWS,
    grid_cols: int = DEFAULT_CONTENT_UPDATE_TILE_GRID_COLS,
    top_fraction: float = DEFAULT_CONTENT_UPDATE_TILE_TOP_FRACTION,
) -> dict[str, float]:
    """Return tile-based change stats that suppress global tone/compression drift.

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
    score = _tile_change_score(
        gray_diff,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        top_fraction=top_fraction,
    )

    return {"score": float(score)}


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


def estimate_change_score_threshold_otsu(
    scores: list[float] | np.ndarray,
    *,
    bins: int = 128,
) -> float:
    """Estimate a binary threshold from all finite change scores using Otsu's method."""
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    if not np.isfinite(value_min) or not np.isfinite(value_max):
        return np.nan
    if value_min >= value_max:
        return value_min

    bin_count = max(2, min(int(bins), int(values.size)))
    hist, edges = np.histogram(values, bins=bin_count, range=(value_min, value_max))
    total = int(np.sum(hist))
    if total <= 0:
        return np.nan

    centers = 0.5 * (edges[:-1] + edges[1:])
    weight_left = np.cumsum(hist).astype(float)
    weight_right = float(total) - weight_left
    sum_left = np.cumsum(hist * centers)
    sum_total = float(sum_left[-1])
    valid = (weight_left > 0.0) & (weight_right > 0.0)
    if not np.any(valid):
        return value_min

    mean_left = sum_left / np.maximum(weight_left, 1e-12)
    mean_right = (sum_total - sum_left) / np.maximum(weight_right, 1e-12)
    between_class_variance = weight_left * weight_right * (mean_left - mean_right) ** 2
    threshold_index = int(np.argmax(np.where(valid, between_class_variance, -np.inf)))
    return float(centers[threshold_index])


def find_update_peaks(
    scores: list[float] | np.ndarray,
    *,
    peak_percentile: float | None = 50.0,
    threshold_method: str = "percentile",
    otsu_bins: int = 128,
    otsu_scale: float = 1.0,
    min_distance_frames: int = 2,
) -> dict[str, Any]:
    """Find local update peaks at or above the configured score threshold."""
    score_array = np.asarray(scores, dtype=float)
    finite_scores = score_array[np.isfinite(score_array)]
    method = str(threshold_method).strip().lower()
    otsu_scale_value = float(otsu_scale) if np.isfinite(otsu_scale) else 1.0
    otsu_scale_value = max(0.0, otsu_scale_value)
    if score_array.size == 0 or finite_scores.size == 0:
        return {
            "peak_indices": [],
            "peak_score_threshold": np.nan,
            "peak_score_percentile": np.nan,
            "peak_threshold_method": method,
            "peak_otsu_bins": int(max(2, int(otsu_bins))),
            "peak_otsu_scale": float(otsu_scale_value),
            "peak_interval_count": 0,
            "peak_threshold_source": "empty_scores",
            "peak_threshold_attempts": 0,
        }

    if method in {"otsu", "auto_otsu"}:
        peak_score_percentile = np.nan
        peak_threshold = estimate_change_score_threshold_otsu(finite_scores, bins=otsu_bins)
        threshold_source = "otsu"
    else:
        peak_score_percentile = float(np.clip(50.0 if peak_percentile is None else peak_percentile, 0.0, 100.0))
        peak_threshold = float(np.percentile(finite_scores, peak_score_percentile))
        threshold_source = "fixed_percentile"
    if not np.isfinite(peak_threshold):
        peak_threshold = float(np.nanmin(finite_scores))
        threshold_source = f"{threshold_source}_fallback_min"
    if method in {"otsu", "auto_otsu"} and np.isfinite(peak_threshold):
        peak_threshold *= otsu_scale_value
        if not np.isclose(otsu_scale_value, 1.0):
            threshold_source = f"{threshold_source}_scaled"
    all_peak_indices = _select_update_peak_indices(
        score_array,
        threshold=peak_threshold,
        min_distance_frames=min_distance_frames,
    )

    return {
        "peak_indices": all_peak_indices,
        "peak_score_threshold": peak_threshold,
        "peak_score_percentile": peak_score_percentile,
        "peak_threshold_method": method,
        "peak_otsu_bins": int(max(2, int(otsu_bins))),
        "peak_otsu_scale": float(otsu_scale_value),
        "peak_interval_count": max(0, len(all_peak_indices) - 1),
        "peak_threshold_source": threshold_source,
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
    continuity_gap_tolerance_frames: float = 2.0,
    scores: list[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Estimate update FPS from the longest continuous peak run."""
    source_fps = float(input_fps) if np.isfinite(input_fps) and float(input_fps) > 0.0 else float(min_fps)
    min_fps_int = max(1, int(min_fps))
    max_fps_int = int(max_fps) if max_fps is not None and int(max_fps) > 0 else int(round(source_fps))
    max_fps_int = max(min_fps_int, max_fps_int)

    peaks = np.asarray(sorted({int(index) for index in np.asarray(peak_indices, dtype=int)}), dtype=int)
    min_peak_count_int = max(2, int(min_peak_count))
    gap_tolerance_frames = float(continuity_gap_tolerance_frames) if np.isfinite(continuity_gap_tolerance_frames) else 2.0
    gap_tolerance_frames = max(0.0, gap_tolerance_frames)
    peak_intervals = np.diff(peaks) if peaks.size >= 2 else np.asarray([], dtype=int)
    positive_peak_intervals = peak_intervals[peak_intervals > 0]
    min_gap_frames = float(np.min(positive_peak_intervals)) if positive_peak_intervals.size else np.nan
    gap_limit_frames = (
        float(min_gap_frames + gap_tolerance_frames)
        if np.isfinite(min_gap_frames)
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
    estimated_output_fps = float(raw_estimated_fps) if np.isfinite(raw_estimated_fps) else float(min_fps_int)
    estimated_output_fps = max(float(min_fps_int), min(float(max_fps_int), estimated_output_fps))
    estimated_fps = int(round(estimated_output_fps)) if np.isfinite(estimated_output_fps) else min_fps_int
    estimated_fps = max(min_fps_int, min(max_fps_int, estimated_fps))
    estimated_interval_frames = max(1, int(round(source_fps / float(estimated_output_fps))))
    return {
        "estimated_fps": int(estimated_fps),
        "estimated_output_fps": float(estimated_output_fps),
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
        "fps_estimation_continuity_gap_tolerance_frames": float(gap_tolerance_frames),
        "fps_estimation_window_score": float(window_score) if np.isfinite(window_score) else np.nan,
        "fps_estimation_window_interval_std": float(window_interval_std) if np.isfinite(window_interval_std) else np.nan,
        "base_interval_raw_frames": float(base_interval_for_fps),
        "base_interval_frames": int(base_interval_frames),
        "base_interval_sec": float(base_interval_for_fps / source_fps) if source_fps > 0.0 else np.nan,
    }


def gap_cycle_from_mean_interval(
    interval_frames: float,
    *,
    max_denominator: int = 30,
) -> list[int]:
    """Build the integer gap cycle implied by a fractional frame interval."""
    if not np.isfinite(interval_frames) or float(interval_frames) <= 0.0:
        return []
    fraction = Fraction(float(interval_frames)).limit_denominator(max(1, int(max_denominator)))
    numerator = max(1, int(fraction.numerator))
    denominator = max(1, int(fraction.denominator))
    if denominator == 1:
        return [max(1, int(round(float(interval_frames))))]
    positions = [(step * numerator) // denominator for step in range(denominator + 1)]
    return [max(1, int(positions[index + 1] - positions[index])) for index in range(denominator)]


def detect_repeating_gap_cycle(
    gaps: list[int] | np.ndarray,
    *,
    max_period: int = 30,
) -> list[int]:
    """Return the shortest exactly repeating gap cycle in a peak run."""
    gap_values = [int(value) for value in np.asarray(gaps, dtype=int) if int(value) > 0]
    if not gap_values:
        return []
    period_limit = min(max(1, int(max_period)), len(gap_values))
    for period in range(1, period_limit + 1):
        if all(int(gap_values[index]) == int(gap_values[index % period]) for index in range(period, len(gap_values))):
            return [int(value) for value in gap_values[:period]]
    return []


def _cycle_residual(gaps: list[int], cycle: list[int]) -> float:
    if not gaps or not cycle:
        return np.inf
    return float(sum(abs(int(gap) - int(cycle[index % len(cycle)])) for index, gap in enumerate(gaps)))


def _best_cycle_rotation_for_gaps(cycle: list[int], gaps: list[int]) -> list[int]:
    if not cycle:
        return []
    best_key: tuple[float, int] | None = None
    best_cycle = [int(value) for value in cycle]
    for offset in range(len(cycle)):
        rotated = [int(cycle[(offset + index) % len(cycle)]) for index in range(len(cycle))]
        key = (_cycle_residual(gaps, rotated), int(offset))
        if best_key is None or key < best_key:
            best_key = key
            best_cycle = rotated
    return best_cycle


def choose_peak_gap_cycle(
    gaps: list[int] | np.ndarray,
    *,
    mean_interval_frames: float,
    max_gap_cycle_period: int = 30,
    max_mean_denominator: int = 30,
) -> dict[str, Any]:
    """Choose a sampling gap cycle from the observed run and mean interval."""
    gap_values = [int(value) for value in np.asarray(gaps, dtype=int) if int(value) > 0]
    run_cycle = detect_repeating_gap_cycle(gap_values, max_period=max_gap_cycle_period)
    mean_cycle = gap_cycle_from_mean_interval(mean_interval_frames, max_denominator=max_mean_denominator)
    mean_cycle = _best_cycle_rotation_for_gaps(mean_cycle, gap_values) if gap_values else mean_cycle

    if run_cycle:
        selected_cycle = run_cycle
        source = "run_gap_cycle"
    elif mean_cycle:
        selected_cycle = mean_cycle
        source = "mean_interval_cycle"
    elif gap_values:
        selected_cycle = [max(1, int(round(float(np.mean(gap_values)))))]
        source = "mean_gap_fixed_interval"
    else:
        selected_cycle = []
        source = "empty"

    return {
        "cycle": [int(value) for value in selected_cycle],
        "cycle_source": source,
        "cycle_period": int(len(selected_cycle)),
        "cycle_mean_frames": float(np.mean(selected_cycle)) if selected_cycle else np.nan,
        "run_gap_cycle": [int(value) for value in run_cycle],
        "mean_interval_cycle": [int(value) for value in mean_cycle],
    }


def expand_indices_from_gap_cycle(
    frame_count: int,
    *,
    anchor_index: int,
    cycle: list[int] | tuple[int, ...],
    next_gap_index: int = 0,
) -> list[int]:
    """Expand a repeating integer gap cycle forward and backward from an anchor."""
    count = max(0, int(frame_count))
    anchor = int(anchor_index)
    gap_cycle = [int(value) for value in cycle if int(value) > 0]
    if count <= 0 or not gap_cycle or not (0 <= anchor < count):
        return []

    indices = {anchor}
    period = len(gap_cycle)

    position = anchor
    gap_index = int(next_gap_index) % period
    while True:
        position += int(gap_cycle[gap_index])
        if position >= count:
            break
        indices.add(int(position))
        gap_index = (gap_index + 1) % period

    position = anchor
    gap_index = (int(next_gap_index) - 1) % period
    while True:
        position -= int(gap_cycle[gap_index])
        if position < 0:
            break
        indices.add(int(position))
        gap_index = (gap_index - 1) % period

    return sorted(indices)


def _peak_interval_selection_metrics(
    selected_indices: list[int] | np.ndarray,
    run_peak_indices: list[int] | np.ndarray,
) -> dict[str, int]:
    """Count selected frames in half-open intervals between consecutive run peaks."""
    run_peaks = np.asarray(sorted({int(index) for index in np.asarray(run_peak_indices, dtype=int)}), dtype=int)
    checked_count = max(0, int(run_peaks.size) - 1)
    if checked_count == 0:
        return {
            "checked_count": 0,
            "violation_count": 0,
            "duplicate_excess_count": 0,
            "missing_count": 0,
            "max_selected_count": 0,
        }

    selected = np.asarray(sorted({int(index) for index in np.asarray(selected_indices, dtype=int)}), dtype=int)
    if selected.size == 0:
        counts = np.zeros(checked_count, dtype=int)
    else:
        counts = np.searchsorted(selected, run_peaks[1:], side="left") - np.searchsorted(selected, run_peaks[:-1], side="left")
    duplicate_counts = counts[counts > 1]
    return {
        "checked_count": int(checked_count),
        "violation_count": int(duplicate_counts.size),
        "duplicate_excess_count": int(np.sum(duplicate_counts - 1)) if duplicate_counts.size else 0,
        "missing_count": int(np.count_nonzero(counts == 0)),
        "max_selected_count": int(np.max(counts)) if counts.size else 0,
    }


def select_indices_by_peak_gap_cycle(
    *,
    frame_count: int,
    peak_indices: list[int] | np.ndarray,
    run_peak_indices: list[int] | np.ndarray,
    mean_interval_frames: float,
    scores: list[float] | np.ndarray | None = None,
    include_first: bool = True,
    include_last: bool = False,
    max_gap_cycle_period: int = 30,
    max_mean_denominator: int = 30,
) -> dict[str, Any]:
    """Select a gap-cycle phase that avoids multiple frames between consecutive peaks."""
    count = max(0, int(frame_count))
    if count == 0:
        return {
            "selected_indices": [],
            "phase_indices": [],
            "selected_phase": 0,
            "selected_phase_score": 0.0,
            "selected_phase_peak_count": 0,
            "selected_phase_recomputed_pair_count": 0,
            "selected_cycle_frames": [],
            "selected_cycle_source": "empty",
            "selected_cycle_period": 0,
            "selected_cycle_mean_frames": np.nan,
            "selected_cycle_anchor_index": -1,
            "selected_cycle_next_gap_index": 0,
            "selected_cycle_run_peak_matches": 0,
            "selected_cycle_selection_mode": "avoid_multiple_frames_between_consecutive_peaks",
            "selected_cycle_peak_gap_checked_count": 0,
            "selected_cycle_peak_gap_violation_count": 0,
            "selected_cycle_peak_gap_duplicate_excess_count": 0,
            "selected_cycle_peak_gap_missing_count": 0,
            "selected_cycle_peak_gap_max_selected_count": 0,
        }

    peak_values = sorted({int(index) for index in np.asarray(peak_indices, dtype=int) if 0 <= int(index) < count})
    run_peak_values = sorted({int(index) for index in np.asarray(run_peak_indices, dtype=int) if 0 <= int(index) < count})
    gaps = np.diff(np.asarray(run_peak_values, dtype=int)) if len(run_peak_values) >= 2 else np.asarray([], dtype=int)
    cycle_info = choose_peak_gap_cycle(
        gaps,
        mean_interval_frames=mean_interval_frames,
        max_gap_cycle_period=max_gap_cycle_period,
        max_mean_denominator=max_mean_denominator,
    )
    cycle = [int(value) for value in cycle_info.get("cycle", []) if int(value) > 0]
    if not cycle:
        fallback_interval = max(1, int(round(float(mean_interval_frames)))) if np.isfinite(mean_interval_frames) else 1
        cycle = [int(fallback_interval)]
        cycle_info = {
            **cycle_info,
            "cycle_source": "fixed_interval_fallback",
            "cycle_period": 1,
            "cycle_mean_frames": float(fallback_interval),
        }

    cycle_span = max(1, int(sum(cycle)))
    anchor_candidates = list(range(min(count, cycle_span)))

    best_key: tuple[int, int, int, int, int, int, int, int] | None = None
    best_phase_indices: list[int] = []
    best_selected_indices: list[int] = []
    best_anchor = int(anchor_candidates[0])
    best_gap_index = 0
    best_peak_count = 0
    best_run_peak_count = 0
    best_metrics = _peak_interval_selection_metrics([], run_peak_values)
    for anchor in anchor_candidates:
        for gap_index in range(len(cycle)):
            phase_indices = expand_indices_from_gap_cycle(
                count,
                anchor_index=int(anchor),
                cycle=cycle,
                next_gap_index=int(gap_index),
            )
            if not phase_indices:
                continue

            candidate_indices = set(phase_indices)
            if include_first:
                candidate_indices.add(0)
            if include_last:
                candidate_indices.add(count - 1)
            candidate_selected_indices = sorted(int(index) for index in candidate_indices if 0 <= int(index) < count)
            candidate_set = set(candidate_selected_indices)
            run_peak_match_count = int(sum(1 for index in run_peak_values if int(index) in candidate_set))
            peak_match_count = int(sum(1 for index in peak_values if int(index) in candidate_set))
            metrics = _peak_interval_selection_metrics(candidate_selected_indices, run_peak_values)
            key = (
                -int(metrics["violation_count"]),
                -int(metrics["duplicate_excess_count"]),
                -int(metrics["missing_count"]),
                -int(metrics["max_selected_count"]),
                run_peak_match_count,
                peak_match_count,
                -int(anchor),
                -int(gap_index),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_phase_indices = [int(index) for index in phase_indices]
                best_selected_indices = candidate_selected_indices
                best_anchor = int(anchor)
                best_gap_index = int(gap_index)
                best_peak_count = peak_match_count
                best_run_peak_count = run_peak_match_count
                best_metrics = metrics

    return {
        "selected_indices": best_selected_indices,
        "phase_indices": best_phase_indices,
        "selected_phase": int(best_anchor),
        "selected_phase_score": float(-best_metrics["duplicate_excess_count"]),
        "selected_phase_peak_count": int(best_peak_count),
        "selected_phase_recomputed_pair_count": 0,
        "selected_cycle_frames": [int(value) for value in cycle],
        "selected_cycle_source": str(cycle_info.get("cycle_source", "unknown")),
        "selected_cycle_period": int(len(cycle)),
        "selected_cycle_mean_frames": float(np.mean(cycle)) if cycle else np.nan,
        "selected_cycle_anchor_index": int(best_anchor),
        "selected_cycle_next_gap_index": int(best_gap_index),
        "selected_cycle_run_peak_matches": int(best_run_peak_count),
        "selected_cycle_selection_mode": "avoid_multiple_frames_between_consecutive_peaks",
        "selected_cycle_peak_gap_checked_count": int(best_metrics["checked_count"]),
        "selected_cycle_peak_gap_violation_count": int(best_metrics["violation_count"]),
        "selected_cycle_peak_gap_duplicate_excess_count": int(best_metrics["duplicate_excess_count"]),
        "selected_cycle_peak_gap_missing_count": int(best_metrics["missing_count"]),
        "selected_cycle_peak_gap_max_selected_count": int(best_metrics["max_selected_count"]),
        "run_gap_cycle_frames": [int(value) for value in cycle_info.get("run_gap_cycle", [])],
        "mean_interval_cycle_frames": [int(value) for value in cycle_info.get("mean_interval_cycle", [])],
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
    }
