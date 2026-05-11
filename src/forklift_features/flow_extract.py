from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd

from . import cache as feature_cache


DEFAULT_DUPLICATE_FRAME_MEAN_ABS_DIFF_THRESHOLD = 0.5
DEFAULT_FLOW_GAP_INTERPOLATION_MAX_SEC = 0.35
DEFAULT_FLOW_GAP_MIN_VECTOR_MAG = 0.5
DEFAULT_FLOW_GAP_DIRECTION_COSINE = 0.70
DEFAULT_FLOW_GAP_MIN_ACTIVE_CELL_RATIO = 0.20
DEFAULT_FLOW_GAP_ACTIVITY_PERCENTILE = 75.0
DEFAULT_FLOW_GAP_BRIDGE_MAX_MISSING_BINS = 3
DEFAULT_FLOW_STATIC_FRAME_MAX_GRID_VECTOR_MAG = 0.02
DEFAULT_FLOW_EDGE_MAG_LOW = 4.0
DEFAULT_FLOW_EDGE_MAG_HIGH = 24.0
DEFAULT_FLOW_EDGE_DENSITY_THRESHOLD = 16.0
DEFAULT_FLOW_EDGE_DENSITY_LOW = 0.005
DEFAULT_FLOW_EDGE_DENSITY_GOOD_LOW = 0.02
DEFAULT_FLOW_EDGE_DENSITY_GOOD_HIGH = 0.25
DEFAULT_FLOW_EDGE_DENSITY_TOO_HIGH = 0.75
DEFAULT_FLOW_EDGE_CLUTTER_CONFIDENCE_FLOOR = 0.20
DEFAULT_FLOW_COHERENCE_MIN_MEAN_MAG = 0.05
DEFAULT_FARNEBACK_PARAMS: dict[str, Any] = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}


def resize_keep_aspect(frame: np.ndarray, width: int | None) -> np.ndarray:
    if width is None or int(width) <= 0:
        return frame
    h, w = frame.shape[:2]
    if w == int(width):
        return frame
    scale = int(width) / max(w, 1)
    return cv2.resize(frame, (int(width), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def video_fps(capture: cv2.VideoCapture) -> float:
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps <= 0.0:
        return 30.0
    return fps


def iter_video_frames(
    video_path: str | Path,
    *,
    resize_width: int | None,
) -> Iterable[tuple[int, float, np.ndarray]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = video_fps(capture)
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield frame_index, float(frame_index / fps), resize_keep_aspect(frame, resize_width)
            frame_index += 1
    finally:
        capture.release()


def _is_missing_path(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def load_frame_time_map(frame_map_path: str | Path | None) -> pd.DataFrame:
    """Load a thinning sidecar mapping output frames to source timestamps."""
    if _is_missing_path(frame_map_path):
        return pd.DataFrame()
    path = Path(frame_map_path)
    if not path.exists():
        return pd.DataFrame()
    frame_map = pd.read_csv(path)
    if frame_map.empty or "output_frame_index" not in frame_map.columns:
        return pd.DataFrame()
    work = frame_map.copy()
    work["output_frame_index"] = pd.to_numeric(work["output_frame_index"], errors="coerce").fillna(-1).astype(int)
    for column in [
        "source_frame_index",
        "source_time_sec",
        "source_video_time_sec",
        "source_duration_sec",
        "gap_capture_frames",
        "gap_sec",
    ]:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    return work.drop_duplicates("output_frame_index").sort_values("output_frame_index").reset_index(drop=True)


def iter_video_frames_with_time_map(
    video_path: str | Path,
    *,
    resize_width: int | None,
    frame_map_path: str | Path | None = None,
) -> Iterable[tuple[int, float, np.ndarray, dict[str, Any]]]:
    """Yield frames using source timestamps from a frame-map sidecar when present."""
    frame_map = load_frame_time_map(frame_map_path)
    metadata_by_output = {
        int(row["output_frame_index"]): dict(row)
        for row in frame_map.to_dict("records")
    } if not frame_map.empty else {}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = video_fps(capture)
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            metadata = dict(metadata_by_output.get(frame_index, {}))
            try:
                source_time = float(metadata.get("source_time_sec", np.nan))
            except (TypeError, ValueError):
                source_time = np.nan
            if not np.isfinite(source_time):
                source_time = float(frame_index / fps)
            metadata.setdefault("output_frame_index", int(frame_index))
            metadata.setdefault("source_frame_index", int(frame_index))
            metadata.setdefault("source_time_sec", float(source_time))
            metadata.setdefault("gap_capture_frames", np.nan)
            metadata.setdefault("gap_sec", np.nan)
            yield frame_index, float(source_time), resize_keep_aspect(frame, resize_width), metadata
            frame_index += 1
    finally:
        capture.release()


def extract_video_frames(
    video_path: str | Path,
    *,
    flow_sample_sec: float,
    resize_width: int | None,
) -> list[dict[str, Any]]:
    del flow_sample_sec
    return [
        {"time": float(time_sec), "frame": frame}
        for _, time_sec, frame in iter_video_frames(video_path, resize_width=resize_width)
    ]


def resize_gray_for_flow(gray: np.ndarray, scale: float) -> tuple[np.ndarray, float, float]:
    scale = float(scale)
    if scale <= 0.0 or scale >= 1.0:
        return gray, 1.0, 1.0
    h, w = gray.shape[:2]
    new_w = max(8, int(round(w * scale)))
    new_h = max(8, int(round(h * scale)))
    if new_w == w and new_h == h:
        return gray, 1.0, 1.0
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, new_w / max(w, 1), new_h / max(h, 1)


def gray_frame_mean_absdiff(previous_gray: np.ndarray, current_gray: np.ndarray) -> float:
    if previous_gray is None or current_gray is None or previous_gray.shape != current_gray.shape:
        return float("inf")
    diff = current_gray.astype(np.float32) - previous_gray.astype(np.float32)
    return float(np.mean(np.abs(diff))) if diff.size else float("inf")


def is_duplicate_gray_frame(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    *,
    mean_abs_diff_threshold: float = DEFAULT_DUPLICATE_FRAME_MEAN_ABS_DIFF_THRESHOLD,
) -> bool:
    threshold = float(mean_abs_diff_threshold)
    if threshold < 0.0:
        return False
    return gray_frame_mean_absdiff(previous_gray, current_gray) <= threshold


def compute_forward_backward_reliability(
    forward_flow: np.ndarray,
    backward_flow: np.ndarray,
    *,
    error_threshold_px: float,
) -> np.ndarray:
    reliable_mask, _ = compute_forward_backward_reliability_and_error(
        forward_flow,
        backward_flow,
        error_threshold_px=error_threshold_px,
    )
    return reliable_mask


def compute_forward_backward_reliability_and_error(
    forward_flow: np.ndarray,
    backward_flow: np.ndarray,
    *,
    error_threshold_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = forward_flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = grid_x + forward_flow[..., 0].astype(np.float32)
    map_y = grid_y + forward_flow[..., 1].astype(np.float32)
    in_bounds = (map_x >= 0.0) & (map_x <= width - 1) & (map_y >= 0.0) & (map_y <= height - 1)
    backward_x = cv2.remap(backward_flow[..., 0].astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    backward_y = cv2.remap(backward_flow[..., 1].astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    error = np.sqrt((forward_flow[..., 0] + backward_x) ** 2 + (forward_flow[..., 1] + backward_y) ** 2)
    reliable_mask = in_bounds & np.isfinite(error) & (error <= float(error_threshold_px))
    return reliable_mask, error.astype(np.float32, copy=False)


def compute_gray_edge_magnitude(gray: np.ndarray) -> np.ndarray:
    gray32 = gray.astype(np.float32, copy=False)
    grad_x = cv2.Sobel(gray32, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray32, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(grad_x, grad_y).astype(np.float32, copy=False)


def edge_confidence_stats(edge_values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(edge_values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0, 0.0
    strength = float(np.percentile(values, 75))
    density = float(np.mean(values >= float(DEFAULT_FLOW_EDGE_DENSITY_THRESHOLD)))
    strength_confidence = np.clip(
        (strength - float(DEFAULT_FLOW_EDGE_MAG_LOW))
        / max(float(DEFAULT_FLOW_EDGE_MAG_HIGH) - float(DEFAULT_FLOW_EDGE_MAG_LOW), 1e-6),
        0.0,
        1.0,
    )
    sparse_confidence = np.clip(
        (density - float(DEFAULT_FLOW_EDGE_DENSITY_LOW))
        / max(float(DEFAULT_FLOW_EDGE_DENSITY_GOOD_LOW) - float(DEFAULT_FLOW_EDGE_DENSITY_LOW), 1e-6),
        0.0,
        1.0,
    )
    clutter_confidence = 1.0 - np.clip(
        (density - float(DEFAULT_FLOW_EDGE_DENSITY_GOOD_HIGH))
        / max(float(DEFAULT_FLOW_EDGE_DENSITY_TOO_HIGH) - float(DEFAULT_FLOW_EDGE_DENSITY_GOOD_HIGH), 1e-6),
        0.0,
        1.0,
    )
    clutter_limited_confidence = (
        float(DEFAULT_FLOW_EDGE_CLUTTER_CONFIDENCE_FLOOR)
        + (1.0 - float(DEFAULT_FLOW_EDGE_CLUTTER_CONFIDENCE_FLOOR)) * clutter_confidence
    )
    density_confidence = sparse_confidence * clutter_limited_confidence
    return strength, density, float(np.sqrt(strength_confidence * density_confidence))


def flow_coherence_confidence(cell_fx: np.ndarray, cell_fy: np.ndarray) -> float:
    fx = np.asarray(cell_fx, dtype=float).reshape(-1)
    fy = np.asarray(cell_fy, dtype=float).reshape(-1)
    finite = np.isfinite(fx) & np.isfinite(fy)
    if not np.any(finite):
        return 0.0
    fx = fx[finite]
    fy = fy[finite]
    magnitudes = np.hypot(fx, fy)
    mean_magnitude = float(np.mean(magnitudes)) if magnitudes.size else 0.0
    if mean_magnitude <= float(DEFAULT_FLOW_COHERENCE_MIN_MEAN_MAG):
        return 1.0
    mean_vector_magnitude = float(np.hypot(np.mean(fx), np.mean(fy)))
    return float(np.clip(mean_vector_magnitude / max(mean_magnitude, 1e-6), 0.0, 1.0))


def grid_bounds(frame_shape: tuple[int, ...], grid: tuple[int, int]) -> Iterable[dict[str, int]]:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    rows, cols = int(grid[0]), int(grid[1])
    for gy in range(rows):
        y0, y1 = int(h * gy / rows), int(h * (gy + 1) / rows)
        for gx in range(cols):
            x0, x1 = int(w * gx / cols), int(w * (gx + 1) / cols)
            yield {"gy": gy, "gx": gx, "x0": x0, "y0": y0, "x1": x1, "y1": y1}


def grid_keys(grid: tuple[int, int]) -> list[tuple[int, int]]:
    return [(gy, gx) for gy in range(int(grid[0])) for gx in range(int(grid[1]))]


def zero_grid_flow_rows(
    frame_shape: tuple[int, ...],
    grid: tuple[int, int],
    *,
    frame_index: int,
    time_sec: float,
    sample_index: int | None,
    reliable_ratio: float = np.nan,
    valid_ratio: float = 1.0,
    flow_mode: str = "zero_hold",
    flow_confidence: float = 1.0,
    observed_dt_sec: float = 0.0,
    source_start_sec: float | None = None,
    source_end_sec: float | None = None,
    gap_capture_frames: float = np.nan,
    flow_interpolated: float = 0.0,
    flow_hold: float = 1.0,
    update_frame_index: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in grid_bounds(frame_shape, grid):
        rows.append({
            "frame_index": int(frame_index),
            "sample_index": int(sample_index) if sample_index is not None else int(frame_index),
            "time_sec": float(time_sec),
            "grid_y": int(cell["gy"]),
            "grid_x": int(cell["gx"]),
            "x0": int(cell["x0"]),
            "y0": int(cell["y0"]),
            "x1": int(cell["x1"]),
            "y1": int(cell["y1"]),
            "center_x": float((cell["x0"] + cell["x1"]) / 2),
            "center_y": float((cell["y0"] + cell["y1"]) / 2),
            "flow_dx_mean": 0.0,
            "flow_dy_mean": 0.0,
            "flow_mag_mean": 0.0,
            "flow_vector_mag": 0.0,
            "flow_reliable_ratio": float(reliable_ratio),
            "flow_backward_consistency": float(reliable_ratio) if np.isfinite(reliable_ratio) else 0.0,
            "valid_ratio": float(valid_ratio),
            "source_dt": 0.0,
            "flow_window_frame_count": 0.0,
            "flow_forward_backward_error_mean": np.nan,
            "flow_edge_strength": 0.0,
            "flow_edge_density": 0.0,
            "flow_edge_confidence": 0.0,
            "flow_coherence_confidence": 1.0,
            "flow_measurement_confidence": 0.0,
            "flow_mode": str(flow_mode),
            "flow_confidence": float(flow_confidence),
            "flow_observed_dt_sec": float(observed_dt_sec),
            "flow_source_start_sec": float(source_start_sec) if source_start_sec is not None else np.nan,
            "flow_source_end_sec": float(source_end_sec) if source_end_sec is not None else float(time_sec),
            "flow_gap_capture_frames": float(gap_capture_frames),
            "flow_interpolated": float(flow_interpolated),
            "flow_hold": float(flow_hold),
            "flow_update_frame_index": int(update_frame_index) if update_frame_index is not None else int(frame_index),
        })
    return rows


def compute_grid_flow_features(
    flow: np.ndarray,
    frame_index: int,
    time_sec: float,
    grid: tuple[int, int],
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    sample_index: int | None = None,
    source_dt: float | None = None,
    reliable_mask: np.ndarray | None = None,
    forward_backward_error: np.ndarray | None = None,
    previous_gray: np.ndarray | None = None,
    current_gray: np.ndarray | None = None,
    flow_mode: str = "observed_update",
    flow_confidence: float = 1.0,
    source_start_sec: float | None = None,
    source_end_sec: float | None = None,
    observed_dt_sec: float | None = None,
    gap_capture_frames: float = np.nan,
    flow_interpolated: float = 0.0,
    flow_hold: float = 0.0,
    update_frame_index: int | None = None,
) -> list[dict[str, Any]]:
    fx = flow[..., 0] / max(float(scale_x), 1e-6)
    fy = flow[..., 1] / max(float(scale_y), 1e-6)
    mag = np.sqrt(fx * fx + fy * fy)
    edge_support = None
    if previous_gray is not None and current_gray is not None and previous_gray.shape[:2] == flow.shape[:2] and current_gray.shape[:2] == flow.shape[:2]:
        previous_edge = compute_gray_edge_magnitude(previous_gray)
        current_edge = compute_gray_edge_magnitude(current_gray)
        edge_support = np.minimum(previous_edge, current_edge)
    rows: list[dict[str, float]] = []
    for cell in grid_bounds(flow.shape[:2], grid):
        x0, y0, x1, y1 = cell["x0"], cell["y0"], cell["x1"], cell["y1"]
        cell_fx = fx[y0:y1, x0:x1]
        cell_fy = fy[y0:y1, x0:x1]
        cell_mag = mag[y0:y1, x0:x1]
        cell_reliable = reliable_mask[y0:y1, x0:x1] if reliable_mask is not None else None
        cell_fb_error = forward_backward_error[y0:y1, x0:x1] if forward_backward_error is not None else None
        cell_edge = edge_support[y0:y1, x0:x1] if edge_support is not None else np.asarray([], dtype=float)
        mean_dx = float(np.mean(cell_fx)) if cell_fx.size else 0.0
        mean_dy = float(np.mean(cell_fy)) if cell_fy.size else 0.0
        finite_fb_error = cell_fb_error[np.isfinite(cell_fb_error)] if cell_fb_error is not None and cell_fb_error.size else np.asarray([], dtype=float)
        edge_strength, edge_density, edge_confidence = edge_confidence_stats(cell_edge)
        coherence_confidence = flow_coherence_confidence(cell_fx, cell_fy)
        reliable_ratio = float(np.mean(cell_reliable)) if cell_reliable is not None and cell_reliable.size else np.nan
        measurement_confidence = float(
            np.clip(
                np.power(
                    max(reliable_ratio if np.isfinite(reliable_ratio) else 0.0, 0.0)
                    * max(edge_confidence, 0.0)
                    * max(coherence_confidence, 0.0),
                    1.0 / 3.0,
                )
                * np.clip(float(flow_confidence), 0.0, 1.0),
                0.0,
                1.0,
            )
        )
        rows.append({
            "frame_index": int(frame_index),
            "sample_index": int(sample_index) if sample_index is not None else int(frame_index),
            "time_sec": float(time_sec),
            "grid_y": int(cell["gy"]),
            "grid_x": int(cell["gx"]),
            "x0": int(x0),
            "y0": int(y0),
            "x1": int(x1),
            "y1": int(y1),
            "center_x": float((x0 + x1) / 2),
            "center_y": float((y0 + y1) / 2),
            "flow_dx_mean": mean_dx,
            "flow_dy_mean": mean_dy,
            "flow_mag_mean": float(np.mean(cell_mag)) if cell_mag.size else 0.0,
            "flow_vector_mag": float(np.hypot(mean_dx, mean_dy)),
            "flow_reliable_ratio": reliable_ratio,
            "flow_backward_consistency": reliable_ratio if np.isfinite(reliable_ratio) else 0.0,
            "valid_ratio": 1.0,
            "source_dt": float(source_dt) if source_dt is not None else np.nan,
            "flow_window_frame_count": 1.0,
            "flow_forward_backward_error_mean": float(np.mean(finite_fb_error)) if finite_fb_error.size else np.nan,
            "flow_edge_strength": edge_strength,
            "flow_edge_density": edge_density,
            "flow_edge_confidence": edge_confidence,
            "flow_coherence_confidence": coherence_confidence,
            "flow_measurement_confidence": measurement_confidence,
            "flow_mode": str(flow_mode),
            "flow_confidence": float(flow_confidence),
            "flow_observed_dt_sec": float(observed_dt_sec) if observed_dt_sec is not None else float(source_dt) if source_dt is not None else np.nan,
            "flow_source_start_sec": float(source_start_sec) if source_start_sec is not None else np.nan,
            "flow_source_end_sec": float(source_end_sec) if source_end_sec is not None else float(time_sec),
            "flow_gap_capture_frames": float(gap_capture_frames),
            "flow_interpolated": float(flow_interpolated),
            "flow_hold": float(flow_hold),
            "flow_update_frame_index": int(update_frame_index) if update_frame_index is not None else int(frame_index),
        })
    return rows


def flow_bin_overlaps(start_sec: float, end_sec: float, bin_sec: float) -> list[tuple[int, float]]:
    width = max(float(bin_sec), 1e-6)
    start = float(start_sec)
    end = float(end_sec)
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return []
    first_bin = int(np.floor(start / width))
    last_bin = int(np.floor((end - min(width * 1e-9, 1e-12)) / width))
    overlaps: list[tuple[int, float]] = []
    for bin_index in range(first_bin, last_bin + 1):
        bin_start = bin_index * width
        bin_end = bin_start + width
        overlap = min(end, bin_end) - max(start, bin_start)
        if overlap > 1e-12:
            overlaps.append((int(bin_index), float(overlap)))
    return overlaps


def _new_grid_flow_accumulator(row: dict[str, Any], *, bin_index: int, bin_sec: float) -> dict[str, Any]:
    return {
        "frame_index": int(row.get("frame_index", 0)),
        "sample_index": int(bin_index),
        "time_sec": round(float(bin_index) * max(float(bin_sec), 1e-6), 10),
        "grid_y": int(row.get("grid_y", row.get("gy", 0))),
        "grid_x": int(row.get("grid_x", row.get("gx", 0))),
        "x0": int(row.get("x0", 0)),
        "y0": int(row.get("y0", 0)),
        "x1": int(row.get("x1", 0)),
        "y1": int(row.get("y1", 0)),
        "center_x": float(row.get("center_x", 0.0)),
        "center_y": float(row.get("center_y", 0.0)),
        "flow_dx_mean": 0.0,
        "flow_dy_mean": 0.0,
        "flow_mag_mean": 0.0,
        "source_dt": 0.0,
        "flow_window_frame_count": 0.0,
        "valid_ratio_sum": 0.0,
        "flow_reliable_ratio_sum": 0.0,
        "flow_reliable_ratio_weight": 0.0,
        "flow_forward_backward_error_sum": 0.0,
        "flow_forward_backward_error_weight": 0.0,
        "flow_edge_strength_sum": 0.0,
        "flow_edge_strength_weight": 0.0,
        "flow_edge_density_sum": 0.0,
        "flow_edge_density_weight": 0.0,
        "flow_edge_confidence_sum": 0.0,
        "flow_edge_confidence_weight": 0.0,
        "flow_coherence_confidence_sum": 0.0,
        "flow_coherence_confidence_weight": 0.0,
        "flow_mode_weights": {},
        "flow_confidence_sum": 0.0,
        "flow_confidence_weight": 0.0,
        "flow_observed_dt_sec_sum": 0.0,
        "flow_observed_dt_sec_weight": 0.0,
        "flow_source_start_sec": np.nan,
        "flow_source_end_sec": np.nan,
        "flow_gap_capture_frames": np.nan,
        "flow_interpolated": 0.0,
        "flow_hold": 0.0,
        "flow_update_frame_index": int(row.get("flow_update_frame_index", row.get("frame_index", 0))),
    }


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _update_grid_flow_accumulator_metadata(acc: dict[str, Any], row: dict[str, Any], weight: float) -> None:
    mode = str(row.get("flow_mode", "observed_update"))
    mode_weights = acc.setdefault("flow_mode_weights", {})
    mode_weights[mode] = float(mode_weights.get(mode, 0.0)) + float(weight)

    confidence = _finite_float(row.get("flow_confidence", np.nan), np.nan)
    if np.isfinite(confidence):
        acc["flow_confidence_sum"] += np.clip(confidence, 0.0, 1.0) * float(weight)
        acc["flow_confidence_weight"] += float(weight)

    observed_dt = _finite_float(row.get("flow_observed_dt_sec", np.nan), np.nan)
    if np.isfinite(observed_dt):
        acc["flow_observed_dt_sec_sum"] += max(observed_dt, 0.0) * float(weight)
        acc["flow_observed_dt_sec_weight"] += float(weight)

    source_start = _finite_float(row.get("flow_source_start_sec", np.nan), np.nan)
    if np.isfinite(source_start):
        current = _finite_float(acc.get("flow_source_start_sec", np.nan), np.nan)
        acc["flow_source_start_sec"] = source_start if not np.isfinite(current) else min(current, source_start)
    source_end = _finite_float(row.get("flow_source_end_sec", np.nan), np.nan)
    if np.isfinite(source_end):
        current = _finite_float(acc.get("flow_source_end_sec", np.nan), np.nan)
        acc["flow_source_end_sec"] = source_end if not np.isfinite(current) else max(current, source_end)

    gap_frames = _finite_float(row.get("flow_gap_capture_frames", np.nan), np.nan)
    if np.isfinite(gap_frames):
        current = _finite_float(acc.get("flow_gap_capture_frames", np.nan), np.nan)
        acc["flow_gap_capture_frames"] = gap_frames if not np.isfinite(current) else max(current, gap_frames)
    acc["flow_interpolated"] = max(float(acc.get("flow_interpolated", 0.0)), _finite_float(row.get("flow_interpolated", 0.0), 0.0))
    acc["flow_hold"] = max(float(acc.get("flow_hold", 0.0)), _finite_float(row.get("flow_hold", 0.0), 0.0))
    acc["flow_update_frame_index"] = max(int(acc.get("flow_update_frame_index", 0)), int(row.get("flow_update_frame_index", row.get("frame_index", 0))))


def accumulate_grid_flow_rows(
    accumulators: dict[tuple[int, int, int], dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    start_sec: float,
    end_sec: float,
    bin_sec: float,
) -> None:
    source_dt = max(float(end_sec) - float(start_sec), 1e-12)
    for bin_index, overlap_sec in flow_bin_overlaps(start_sec, end_sec, bin_sec):
        fraction = float(overlap_sec) / source_dt
        for row in rows:
            gy = int(row.get("grid_y", row.get("gy", 0)))
            gx = int(row.get("grid_x", row.get("gx", 0)))
            key = (int(bin_index), gy, gx)
            acc = accumulators.get(key)
            if acc is None:
                acc = _new_grid_flow_accumulator(row, bin_index=int(bin_index), bin_sec=bin_sec)
                accumulators[key] = acc
            acc["frame_index"] = max(int(acc["frame_index"]), int(row.get("frame_index", 0)))
            acc["flow_dx_mean"] += _finite_float(row.get("flow_dx_mean", 0.0)) * fraction
            acc["flow_dy_mean"] += _finite_float(row.get("flow_dy_mean", 0.0)) * fraction
            acc["flow_mag_mean"] += _finite_float(row.get("flow_mag_mean", 0.0)) * fraction
            acc["source_dt"] += float(overlap_sec)
            acc["flow_window_frame_count"] += fraction
            acc["valid_ratio_sum"] += np.clip(_finite_float(row.get("valid_ratio", 1.0), 1.0), 0.0, 1.0) * float(overlap_sec)
            reliable_value = _finite_float(row.get("flow_reliable_ratio", np.nan), np.nan)
            if np.isfinite(reliable_value):
                acc["flow_reliable_ratio_sum"] += np.clip(reliable_value, 0.0, 1.0) * float(overlap_sec)
                acc["flow_reliable_ratio_weight"] += float(overlap_sec)
            error_value = _finite_float(row.get("flow_forward_backward_error_mean", np.nan), np.nan)
            if np.isfinite(error_value):
                acc["flow_forward_backward_error_sum"] += max(error_value, 0.0) * float(overlap_sec)
                acc["flow_forward_backward_error_weight"] += float(overlap_sec)
            for row_col, sum_col, weight_col in [
                ("flow_edge_strength", "flow_edge_strength_sum", "flow_edge_strength_weight"),
                ("flow_edge_density", "flow_edge_density_sum", "flow_edge_density_weight"),
                ("flow_edge_confidence", "flow_edge_confidence_sum", "flow_edge_confidence_weight"),
                ("flow_coherence_confidence", "flow_coherence_confidence_sum", "flow_coherence_confidence_weight"),
            ]:
                value = _finite_float(row.get(row_col, np.nan), np.nan)
                if np.isfinite(value):
                    acc[sum_col] += max(value, 0.0) * float(overlap_sec)
                    acc[weight_col] += float(overlap_sec)
            _update_grid_flow_accumulator_metadata(acc, row, float(overlap_sec))


def accumulate_grid_flow_rows_to_bin(
    accumulators: dict[tuple[int, int, int], dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    bin_index: int,
    bin_sec: float,
) -> None:
    weight = max(float(bin_sec), 1e-6)
    for row in rows:
        gy = int(row.get("grid_y", row.get("gy", 0)))
        gx = int(row.get("grid_x", row.get("gx", 0)))
        key = (int(bin_index), gy, gx)
        acc = accumulators.get(key)
        if acc is None:
            acc = _new_grid_flow_accumulator(row, bin_index=int(bin_index), bin_sec=bin_sec)
            accumulators[key] = acc
        acc["frame_index"] = max(int(acc["frame_index"]), int(row.get("frame_index", 0)))
        acc["flow_dx_mean"] += _finite_float(row.get("flow_dx_mean", 0.0))
        acc["flow_dy_mean"] += _finite_float(row.get("flow_dy_mean", 0.0))
        acc["flow_mag_mean"] += _finite_float(row.get("flow_mag_mean", 0.0))
        acc["source_dt"] += max(_finite_float(row.get("source_dt", 0.0), 0.0), 0.0)
        acc["flow_window_frame_count"] += _finite_float(row.get("flow_window_frame_count", 1.0), 1.0)
        acc["valid_ratio_sum"] += np.clip(_finite_float(row.get("valid_ratio", 1.0), 1.0), 0.0, 1.0) * weight
        reliable_value = _finite_float(row.get("flow_reliable_ratio", np.nan), np.nan)
        if np.isfinite(reliable_value):
            acc["flow_reliable_ratio_sum"] += np.clip(reliable_value, 0.0, 1.0) * weight
            acc["flow_reliable_ratio_weight"] += weight
        error_value = _finite_float(row.get("flow_forward_backward_error_mean", np.nan), np.nan)
        if np.isfinite(error_value):
            acc["flow_forward_backward_error_sum"] += max(error_value, 0.0) * weight
            acc["flow_forward_backward_error_weight"] += weight
        for row_col, sum_col, weight_col in [
            ("flow_edge_strength", "flow_edge_strength_sum", "flow_edge_strength_weight"),
            ("flow_edge_density", "flow_edge_density_sum", "flow_edge_density_weight"),
            ("flow_edge_confidence", "flow_edge_confidence_sum", "flow_edge_confidence_weight"),
            ("flow_coherence_confidence", "flow_coherence_confidence_sum", "flow_coherence_confidence_weight"),
        ]:
            value = _finite_float(row.get(row_col, np.nan), np.nan)
            if np.isfinite(value):
                acc[sum_col] += max(value, 0.0) * weight
                acc[weight_col] += weight
        _update_grid_flow_accumulator_metadata(acc, row, weight)


def finalize_grid_flow_window_rows(accumulators: dict[tuple[int, int, int], dict[str, Any]], *, bin_sec: float) -> list[dict[str, float]]:
    width = max(float(bin_sec), 1e-6)
    rows: list[dict[str, float]] = []
    for _, acc in sorted(accumulators.items(), key=lambda item: item[0]):
        covered_sec = max(float(acc.get("source_dt", 0.0)), 0.0)
        reliable_weight = float(acc.get("flow_reliable_ratio_weight", 0.0))
        error_weight = float(acc.get("flow_forward_backward_error_weight", 0.0))
        mean_dx = float(acc["flow_dx_mean"])
        mean_dy = float(acc["flow_dy_mean"])
        mode_weights = acc.get("flow_mode_weights", {})
        flow_mode = max(mode_weights.items(), key=lambda item: item[1])[0] if mode_weights else "observed_update"
        confidence_weight = float(acc.get("flow_confidence_weight", 0.0))
        observed_dt_weight = float(acc.get("flow_observed_dt_sec_weight", 0.0))
        edge_strength_weight = float(acc.get("flow_edge_strength_weight", 0.0))
        edge_density_weight = float(acc.get("flow_edge_density_weight", 0.0))
        edge_confidence_weight = float(acc.get("flow_edge_confidence_weight", 0.0))
        coherence_confidence_weight = float(acc.get("flow_coherence_confidence_weight", 0.0))
        reliable_ratio = float(acc["flow_reliable_ratio_sum"] / reliable_weight) if reliable_weight > 0.0 else 0.0
        valid_ratio = float(np.clip(acc["valid_ratio_sum"] / width, 0.0, 1.0))
        flow_confidence = float(acc["flow_confidence_sum"] / confidence_weight) if confidence_weight > 0.0 else 1.0
        edge_confidence = float(acc["flow_edge_confidence_sum"] / edge_confidence_weight) if edge_confidence_weight > 0.0 else 0.0
        coherence_confidence = float(acc["flow_coherence_confidence_sum"] / coherence_confidence_weight) if coherence_confidence_weight > 0.0 else 1.0
        measurement_confidence = float(np.clip(
            flow_confidence
            * valid_ratio
            * np.power(max(reliable_ratio, 0.0) * max(edge_confidence, 0.0) * max(coherence_confidence, 0.0), 1.0 / 3.0),
            0.0,
            1.0,
        ))
        rows.append({
            "frame_index": int(acc["frame_index"]),
            "sample_index": int(acc["sample_index"]),
            "time_sec": float(acc["time_sec"]),
            "grid_y": int(acc["grid_y"]),
            "grid_x": int(acc["grid_x"]),
            "x0": int(acc["x0"]),
            "y0": int(acc["y0"]),
            "x1": int(acc["x1"]),
            "y1": int(acc["y1"]),
            "center_x": float(acc["center_x"]),
            "center_y": float(acc["center_y"]),
            "flow_dx_mean": mean_dx,
            "flow_dy_mean": mean_dy,
            "flow_mag_mean": float(acc["flow_mag_mean"]),
            "flow_vector_mag": float(np.hypot(mean_dx, mean_dy)),
            "flow_reliable_ratio": reliable_ratio,
            "flow_backward_consistency": reliable_ratio,
            "valid_ratio": valid_ratio,
            "source_dt": covered_sec,
            "flow_window_frame_count": float(acc["flow_window_frame_count"]),
            "flow_forward_backward_error_mean": float(acc["flow_forward_backward_error_sum"] / error_weight) if error_weight > 0.0 else np.nan,
            "flow_edge_strength": float(acc["flow_edge_strength_sum"] / edge_strength_weight) if edge_strength_weight > 0.0 else 0.0,
            "flow_edge_density": float(acc["flow_edge_density_sum"] / edge_density_weight) if edge_density_weight > 0.0 else 0.0,
            "flow_edge_confidence": edge_confidence,
            "flow_coherence_confidence": coherence_confidence,
            "flow_measurement_confidence": measurement_confidence,
            "flow_mode": str(flow_mode),
            "flow_confidence": flow_confidence,
            "flow_observed_dt_sec": float(acc["flow_observed_dt_sec_sum"] / observed_dt_weight) if observed_dt_weight > 0.0 else np.nan,
            "flow_source_start_sec": float(acc.get("flow_source_start_sec", np.nan)),
            "flow_source_end_sec": float(acc.get("flow_source_end_sec", np.nan)),
            "flow_gap_capture_frames": float(acc.get("flow_gap_capture_frames", np.nan)),
            "flow_interpolated": float(acc.get("flow_interpolated", 0.0)),
            "flow_hold": float(acc.get("flow_hold", 0.0)),
            "flow_update_frame_index": int(acc.get("flow_update_frame_index", acc["frame_index"])),
        })
    return rows


def compute_grid_flow_from_frame_iter(
    frame_iter: Iterable[tuple[int, float, np.ndarray]],
    *,
    flow_sample_sec: float,
    flow_analysis_scale: float,
    flow_grid: tuple[int, int],
    reliable_error_threshold_px: float,
) -> pd.DataFrame:
    sample_sec = max(float(flow_sample_sec), 1e-6)
    iterator = iter(frame_iter)
    try:
        first_frame_index, first_time_sec, first_frame = next(iterator)
    except StopIteration:
        return pd.DataFrame()

    prev_gray, _, _ = resize_gray_for_flow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY), flow_analysis_scale)
    prev_time = float(first_time_sec)
    frame_shape = prev_gray.shape[:2]
    accumulators: dict[tuple[int, int, int], dict[str, Any]] = {}

    for frame_index, current_time_sec, frame in iterator:
        current_time = float(current_time_sec)
        source_start = prev_time
        source_end = current_time
        if not np.isfinite(source_end) or source_end <= source_start:
            source_end = source_start + sample_sec
        source_dt = max(source_end - source_start, 1e-12)
        try:
            gray, scale_x, scale_y = resize_gray_for_flow(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), flow_analysis_scale)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **DEFAULT_FARNEBACK_PARAMS)
            backward = cv2.calcOpticalFlowFarneback(gray, prev_gray, None, **DEFAULT_FARNEBACK_PARAMS)
            reliable_mask, forward_backward_error = compute_forward_backward_reliability_and_error(
                flow,
                backward,
                error_threshold_px=reliable_error_threshold_px,
            )
            frame_grid_features = compute_grid_flow_features(
                flow,
                int(frame_index),
                current_time,
                flow_grid,
                sample_index=max(0, int(np.floor(source_start / sample_sec))),
                source_dt=source_dt,
                scale_x=scale_x,
                scale_y=scale_y,
                reliable_mask=reliable_mask,
                forward_backward_error=forward_backward_error,
                previous_gray=prev_gray,
                current_gray=gray,
            )
        except Exception:
            gray, _, _ = resize_gray_for_flow(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), flow_analysis_scale)
            frame_grid_features = zero_grid_flow_rows(
                gray.shape[:2],
                flow_grid,
                frame_index=int(frame_index),
                time_sec=current_time,
                sample_index=max(0, int(np.floor(source_start / sample_sec))),
                reliable_ratio=0.0,
            )
            for row in frame_grid_features:
                row["source_dt"] = source_dt
        accumulate_grid_flow_rows(
            accumulators,
            frame_grid_features,
            start_sec=source_start,
            end_sec=source_end,
            bin_sec=sample_sec,
        )
        prev_gray = gray
        prev_time = source_end

    grid_rows = finalize_grid_flow_window_rows(accumulators, bin_sec=sample_sec)
    if not grid_rows:
        grid_rows = zero_grid_flow_rows(
            frame_shape,
            flow_grid,
            frame_index=int(first_frame_index),
            time_sec=float(first_time_sec),
            sample_index=0,
            reliable_ratio=0.0,
        )
        for row in grid_rows:
            row["valid_ratio"] = 0.0
    return pd.DataFrame(grid_rows).sort_values(["sample_index", "grid_y", "grid_x"]).reset_index(drop=True)


def _frame_item_parts(item: Any) -> tuple[int, float, np.ndarray, dict[str, Any]]:
    if len(item) >= 4:
        frame_index, time_sec, frame, metadata = item[:4]
        return int(frame_index), float(time_sec), frame, dict(metadata or {})
    frame_index, time_sec, frame = item[:3]
    return int(frame_index), float(time_sec), frame, {}


def _grid_flow_row_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row.get("grid_y", row.get("gy", 0))), int(row.get("grid_x", row.get("gx", 0)))


def _grid_flow_row_vector_mag(row: dict[str, Any]) -> float:
    mag = _finite_float(row.get("flow_vector_mag", np.nan), np.nan)
    if np.isfinite(mag):
        return max(float(mag), 0.0)
    return float(np.hypot(
        _finite_float(row.get("flow_dx_mean", 0.0), 0.0),
        _finite_float(row.get("flow_dy_mean", 0.0), 0.0),
    ))


def _grid_flow_rows_are_static(
    rows: list[dict[str, Any]],
    *,
    max_grid_vector_mag: float = DEFAULT_FLOW_STATIC_FRAME_MAX_GRID_VECTOR_MAG,
) -> bool:
    """Return True when every grid cell is nearly motionless."""
    if not rows:
        return False
    threshold = max(float(max_grid_vector_mag), 0.0)
    return all(_grid_flow_row_vector_mag(row) <= threshold for row in rows)


def _assign_scaled_flow_from_source(
    target: dict[str, Any],
    source: dict[str, Any],
    *,
    fraction: float,
    flow_mode: str,
    flow_interpolated: float,
) -> None:
    dx = _finite_float(source.get("flow_dx_mean", 0.0), 0.0) * float(fraction)
    dy = _finite_float(source.get("flow_dy_mean", 0.0), 0.0) * float(fraction)
    source_mag = _finite_float(source.get("flow_mag_mean", np.nan), np.nan)
    target["flow_dx_mean"] = float(dx)
    target["flow_dy_mean"] = float(dy)
    target["flow_vector_mag"] = float(np.hypot(dx, dy))
    target["flow_mag_mean"] = float(abs(float(fraction)) * source_mag) if np.isfinite(source_mag) else float(target["flow_vector_mag"])
    for column in [
        "valid_ratio",
        "flow_reliable_ratio",
        "flow_backward_consistency",
        "flow_forward_backward_error_mean",
        "flow_edge_strength",
        "flow_edge_density",
        "flow_edge_confidence",
        "flow_coherence_confidence",
        "flow_measurement_confidence",
        "flow_confidence",
    ]:
        if column in source:
            target[column] = source[column]
    target["flow_mode"] = str(flow_mode)
    target["flow_interpolated"] = float(flow_interpolated)
    target["flow_hold"] = 0.0


def apportion_isolated_static_frame_flow_intervals(
    intervals: list[dict[str, Any]],
    *,
    max_grid_vector_mag: float = DEFAULT_FLOW_STATIC_FRAME_MAX_GRID_VECTOR_MAG,
) -> list[dict[str, Any]]:
    """Split the following measured flow across an isolated static frame.

    Each interval represents previous_frame -> current_frame. When the current
    frame is static but the previous and following frames are not static, the
    following interval's flow is split equally into the static interval and the
    following interval.
    """
    if len(intervals) < 3:
        return intervals

    corrected = [
        {
            **interval,
            "rows": [dict(row) for row in interval.get("rows", [])],
        }
        for interval in intervals
    ]
    static_flags = [
        _grid_flow_rows_are_static(interval.get("rows", []), max_grid_vector_mag=max_grid_vector_mag)
        for interval in intervals
    ]
    for index in range(1, len(intervals) - 1):
        if not (static_flags[index] and not static_flags[index - 1] and not static_flags[index + 1]):
            continue
        measured_next_by_grid = {
            _grid_flow_row_key(row): row
            for row in intervals[index + 1].get("rows", [])
        }
        for row in corrected[index].get("rows", []):
            source = measured_next_by_grid.get(_grid_flow_row_key(row))
            if source is None:
                continue
            _assign_scaled_flow_from_source(
                row,
                source,
                fraction=0.5,
                flow_mode="isolated_static_frame_split_previous",
                flow_interpolated=1.0,
            )
        for row in corrected[index + 1].get("rows", []):
            source = measured_next_by_grid.get(_grid_flow_row_key(row))
            if source is None:
                continue
            _assign_scaled_flow_from_source(
                row,
                source,
                fraction=0.5,
                flow_mode="isolated_static_frame_split_next",
                flow_interpolated=_finite_float(source.get("flow_interpolated", 0.0), 0.0),
            )
    return corrected


def compute_static_frame_split_grid_flow_from_frame_iter(
    frame_iter: Iterable[Any],
    *,
    flow_sample_sec: float,
    flow_analysis_scale: float,
    flow_grid: tuple[int, int],
    reliable_error_threshold_px: float,
    static_frame_max_grid_vector_mag: float = DEFAULT_FLOW_STATIC_FRAME_MAX_GRID_VECTOR_MAG,
) -> pd.DataFrame:
    sample_sec = max(float(flow_sample_sec), 1e-6)
    iterator = iter(frame_iter)
    try:
        first_item = next(iterator)
    except StopIteration:
        return pd.DataFrame()
    first_frame_index, first_time_sec, first_frame, first_metadata = _frame_item_parts(first_item)

    prev_gray, _, _ = resize_gray_for_flow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY), flow_analysis_scale)
    prev_time = float(first_time_sec)
    prev_frame_index = int(first_frame_index)
    frame_shape = prev_gray.shape[:2]
    duration_candidates = [_finite_float(first_metadata.get("source_duration_sec", np.nan), np.nan)]
    intervals: list[dict[str, Any]] = []

    for item in iterator:
        frame_index, current_time_sec, frame, metadata = _frame_item_parts(item)
        current_time = float(current_time_sec)
        source_start = prev_time
        source_end = current_time
        if not np.isfinite(source_end) or source_end <= source_start:
            source_end = source_start + sample_sec
        source_dt = max(source_end - source_start, 1e-12)
        gap_capture_frames = _finite_float(metadata.get("gap_capture_frames", frame_index - prev_frame_index), np.nan)
        duration_candidates.append(_finite_float(metadata.get("source_duration_sec", np.nan), np.nan))
        try:
            gray, scale_x, scale_y = resize_gray_for_flow(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), flow_analysis_scale)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **DEFAULT_FARNEBACK_PARAMS)
            backward = cv2.calcOpticalFlowFarneback(gray, prev_gray, None, **DEFAULT_FARNEBACK_PARAMS)
            reliable_mask, forward_backward_error = compute_forward_backward_reliability_and_error(
                flow,
                backward,
                error_threshold_px=reliable_error_threshold_px,
            )
            frame_grid_features = compute_grid_flow_features(
                flow,
                int(frame_index),
                current_time,
                flow_grid,
                sample_index=max(0, int(np.floor(source_start / sample_sec))),
                source_dt=source_dt,
                scale_x=scale_x,
                scale_y=scale_y,
                reliable_mask=reliable_mask,
                forward_backward_error=forward_backward_error,
                previous_gray=prev_gray,
                current_gray=gray,
                source_start_sec=source_start,
                source_end_sec=source_end,
                observed_dt_sec=source_dt,
                gap_capture_frames=gap_capture_frames,
                update_frame_index=int(metadata.get("source_frame_index", frame_index)),
            )
        except Exception:
            gray, _, _ = resize_gray_for_flow(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), flow_analysis_scale)
            frame_grid_features = zero_grid_flow_rows(
                gray.shape[:2],
                flow_grid,
                frame_index=int(frame_index),
                time_sec=current_time,
                sample_index=max(0, int(np.floor(source_start / sample_sec))),
                reliable_ratio=0.0,
                flow_mode="flow_failed",
                flow_confidence=0.0,
                observed_dt_sec=source_dt,
                source_start_sec=source_start,
                source_end_sec=source_end,
                gap_capture_frames=gap_capture_frames,
                flow_hold=0.0,
                update_frame_index=int(metadata.get("source_frame_index", frame_index)),
            )
            for row in frame_grid_features:
                row["source_dt"] = source_dt
        intervals.append({
            "start_sec": float(source_start),
            "end_sec": float(source_end),
            "rows": frame_grid_features,
        })
        prev_gray = gray
        prev_time = source_end
        prev_frame_index = int(frame_index)

    intervals = apportion_isolated_static_frame_flow_intervals(
        intervals,
        max_grid_vector_mag=static_frame_max_grid_vector_mag,
    )
    accumulators: dict[tuple[int, int, int], dict[str, Any]] = {}
    for interval in intervals:
        accumulate_grid_flow_rows(
            accumulators,
            interval.get("rows", []),
            start_sec=float(interval.get("start_sec", 0.0)),
            end_sec=float(interval.get("end_sec", 0.0)),
            bin_sec=sample_sec,
        )

    finite_durations = [value for value in duration_candidates if np.isfinite(value) and value > 0.0]
    if finite_durations:
        complete_grid_flow_bins(
            accumulators,
            frame_shape,
            flow_grid,
            duration_sec=max(finite_durations),
            bin_sec=sample_sec,
        )
    grid_rows = finalize_grid_flow_window_rows(accumulators, bin_sec=sample_sec)
    if not grid_rows:
        grid_rows = zero_grid_flow_rows(
            frame_shape,
            flow_grid,
            frame_index=int(first_frame_index),
            time_sec=float(first_time_sec),
            sample_index=0,
            reliable_ratio=0.0,
        )
        for row in grid_rows:
            row["valid_ratio"] = 0.0
    return pd.DataFrame(grid_rows).sort_values(["sample_index", "grid_y", "grid_x"]).reset_index(drop=True)


def _event_bin_index(time_sec: float, bin_sec: float) -> int:
    width = max(float(bin_sec), 1e-6)
    epsilon = min(width * 1e-6, 1e-9)
    return max(0, int(np.floor((max(float(time_sec), 0.0) - epsilon) / width)))


def _grid_flow_motion_summary(
    rows: list[dict[str, Any]],
    *,
    active_mag_threshold: float,
    activity_percentile: float = DEFAULT_FLOW_GAP_ACTIVITY_PERCENTILE,
) -> dict[str, float]:
    if not rows:
        return {"strength": 0.0, "active_ratio": 0.0, "direction_x": 0.0, "direction_y": 0.0, "direction_confidence": 0.0}
    dx = np.asarray([_finite_float(row.get("flow_dx_mean", 0.0), 0.0) for row in rows], dtype=float)
    dy = np.asarray([_finite_float(row.get("flow_dy_mean", 0.0), 0.0) for row in rows], dtype=float)
    mag = np.asarray([
        _finite_float(row.get("flow_vector_mag", np.nan), np.nan)
        for row in rows
    ], dtype=float)
    fallback_mag = np.hypot(dx, dy)
    mag = np.where(np.isfinite(mag), mag, fallback_mag)
    finite = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(mag)
    if not np.any(finite):
        return {"strength": 0.0, "active_ratio": 0.0, "direction_x": 0.0, "direction_y": 0.0, "direction_confidence": 0.0}
    dx = dx[finite]
    dy = dy[finite]
    mag = np.clip(mag[finite], 0.0, None)
    strength = float(np.percentile(mag, np.clip(float(activity_percentile), 0.0, 100.0))) if mag.size else 0.0
    active = mag >= max(float(active_mag_threshold), 0.0)
    active_ratio = float(np.mean(active)) if mag.size else 0.0
    if not np.any(active):
        return {"strength": strength, "active_ratio": active_ratio, "direction_x": 0.0, "direction_y": 0.0, "direction_confidence": 0.0}
    active_mag = np.maximum(mag[active], 1e-6)
    unit_x = dx[active] / active_mag
    unit_y = dy[active] / active_mag
    weights = active_mag / max(float(np.sum(active_mag)), 1e-6)
    direction_x = float(np.sum(weights * unit_x))
    direction_y = float(np.sum(weights * unit_y))
    direction_confidence = float(np.clip(np.hypot(direction_x, direction_y), 0.0, 1.0))
    return {
        "strength": strength,
        "active_ratio": active_ratio,
        "direction_x": direction_x,
        "direction_y": direction_y,
        "direction_confidence": direction_confidence,
    }


def _motion_summary_has_activity(
    summary: dict[str, float] | None,
    *,
    min_vector_mag: float,
    min_active_cell_ratio: float,
) -> bool:
    if summary is None:
        return False
    return bool(
        _finite_float(summary.get("strength", 0.0), 0.0) >= float(min_vector_mag)
        or _finite_float(summary.get("active_ratio", 0.0), 0.0) >= float(min_active_cell_ratio)
    )


def _should_interpolate_motion_gap(
    previous_summary: dict[str, float] | None,
    current_summary: dict[str, float],
    *,
    source_dt: float,
    max_gap_sec: float,
    min_vector_mag: float,
    min_active_cell_ratio: float,
    direction_cosine_threshold: float,
) -> tuple[bool, bool]:
    if previous_summary is None:
        current_active = _motion_summary_has_activity(
            current_summary,
            min_vector_mag=min_vector_mag,
            min_active_cell_ratio=min_active_cell_ratio,
        )
        return current_active, False
    previous_active = _motion_summary_has_activity(
        previous_summary,
        min_vector_mag=min_vector_mag,
        min_active_cell_ratio=min_active_cell_ratio,
    )
    current_active = _motion_summary_has_activity(
        current_summary,
        min_vector_mag=min_vector_mag,
        min_active_cell_ratio=min_active_cell_ratio,
    )
    motion_active = bool(current_active)
    if source_dt > float(max_gap_sec) or not (previous_active and current_active):
        return motion_active, False
    prev = np.asarray([
        _finite_float(previous_summary.get("direction_x", 0.0), 0.0),
        _finite_float(previous_summary.get("direction_y", 0.0), 0.0),
    ], dtype=float)
    curr = np.asarray([
        _finite_float(current_summary.get("direction_x", 0.0), 0.0),
        _finite_float(current_summary.get("direction_y", 0.0), 0.0),
    ], dtype=float)
    prev_mag = float(np.linalg.norm(prev))
    curr_mag = float(np.linalg.norm(curr))
    if prev_mag <= 1e-6 or curr_mag <= 1e-6:
        return True, False
    cosine = float(np.dot(prev, curr) / max(prev_mag * curr_mag, 1e-12))
    return True, bool(np.isfinite(cosine) and cosine >= float(direction_cosine_threshold))


def complete_grid_flow_bins(
    accumulators: dict[tuple[int, int, int], dict[str, Any]],
    frame_shape: tuple[int, ...],
    grid: tuple[int, int],
    *,
    duration_sec: float,
    bin_sec: float,
) -> None:
    if not np.isfinite(duration_sec) or duration_sec <= 0.0:
        return
    last_bin = max(0, int(np.ceil(float(duration_sec) / max(float(bin_sec), 1e-6))) - 1)
    existing_bins = {int(key[0]) for key in accumulators}
    for bin_index in range(last_bin + 1):
        if bin_index in existing_bins:
            continue
        time_sec = float(bin_index) * max(float(bin_sec), 1e-6)
        rows = zero_grid_flow_rows(
            frame_shape,
            grid,
            frame_index=0,
            time_sec=time_sec,
            sample_index=bin_index,
            reliable_ratio=1.0,
            valid_ratio=1.0,
            flow_mode="zero_hold",
            flow_confidence=0.95,
            observed_dt_sec=0.0,
            source_start_sec=time_sec,
            source_end_sec=min(time_sec + float(bin_sec), float(duration_sec)),
            flow_interpolated=0.0,
            flow_hold=1.0,
            update_frame_index=0,
        )
        accumulate_grid_flow_rows_to_bin(accumulators, rows, bin_index=bin_index, bin_sec=bin_sec)


def bridge_isolated_motion_gap_rows(
    grid_rows: list[dict[str, Any]],
    *,
    min_vector_mag: float = DEFAULT_FLOW_GAP_MIN_VECTOR_MAG,
    max_missing_bins: int = DEFAULT_FLOW_GAP_BRIDGE_MAX_MISSING_BINS,
) -> list[dict[str, Any]]:
    if not grid_rows:
        return grid_rows
    work = pd.DataFrame(grid_rows)
    required = {"sample_index", "grid_y", "grid_x", "flow_dx_mean", "flow_dy_mean", "flow_vector_mag", "flow_mode"}
    if work.empty or required - set(work.columns):
        return grid_rows
    work = work.sort_values(["grid_y", "grid_x", "sample_index"]).reset_index(drop=True)
    bridge_indices: list[int] = []
    for _, group in work.groupby(["grid_y", "grid_x"], sort=False):
        group_indices = list(group.index)
        pos = 0
        while pos < len(group_indices):
            idx = group_indices[pos]
            row = work.loc[idx]
            is_missing = (
                str(row.get("flow_mode", "")) == "zero_hold"
                and _finite_float(row.get("flow_vector_mag", 0.0), 0.0) < float(min_vector_mag)
            )
            if not is_missing:
                pos += 1
                continue
            run_start = pos
            while pos < len(group_indices):
                run_idx = group_indices[pos]
                run_row = work.loc[run_idx]
                if not (
                    str(run_row.get("flow_mode", "")) == "zero_hold"
                    and _finite_float(run_row.get("flow_vector_mag", 0.0), 0.0) < float(min_vector_mag)
                ):
                    break
                pos += 1
            run_end = pos
            run_len = run_end - run_start
            if run_len <= 0 or run_len > int(max_missing_bins) or run_start == 0 or run_end >= len(group_indices):
                continue
            prev_idx = group_indices[run_start - 1]
            next_idx = group_indices[run_end]
            prev_row = work.loc[prev_idx]
            next_row = work.loc[next_idx]
            prev_mag = _finite_float(prev_row.get("flow_vector_mag", 0.0), 0.0)
            next_mag = _finite_float(next_row.get("flow_vector_mag", 0.0), 0.0)
            if prev_mag < float(min_vector_mag) or next_mag < float(min_vector_mag):
                continue
            prev_bin = int(prev_row.get("sample_index", -1))
            next_bin = int(next_row.get("sample_index", -1))
            run_bins = [int(work.at[group_indices[i], "sample_index"]) for i in range(run_start, run_end)]
            if run_bins != list(range(prev_bin + 1, next_bin)):
                continue
            for offset, run_idx in enumerate(group_indices[run_start:run_end], start=1):
                fraction = float(offset) / float(run_len + 1)
                dx = (1.0 - fraction) * _finite_float(prev_row.get("flow_dx_mean", 0.0), 0.0) + fraction * _finite_float(next_row.get("flow_dx_mean", 0.0), 0.0)
                dy = (1.0 - fraction) * _finite_float(prev_row.get("flow_dy_mean", 0.0), 0.0) + fraction * _finite_float(next_row.get("flow_dy_mean", 0.0), 0.0)
                mag = float(np.hypot(dx, dy))
                work.at[run_idx, "flow_dx_mean"] = dx
                work.at[run_idx, "flow_dy_mean"] = dy
                work.at[run_idx, "flow_mag_mean"] = mag
                work.at[run_idx, "flow_vector_mag"] = mag
                work.at[run_idx, "flow_mode"] = "bridged_motion_gap"
                work.at[run_idx, "flow_confidence"] = min(
                    0.25,
                    max(
                        0.05,
                        0.5 * min(
                            _finite_float(prev_row.get("flow_confidence", 0.0), 0.0),
                            _finite_float(next_row.get("flow_confidence", 0.0), 0.0),
                        ),
                    ),
                )
                work.at[run_idx, "flow_interpolated"] = 1.0
                work.at[run_idx, "flow_hold"] = 0.0
                work.at[run_idx, "flow_backward_consistency"] = min(
                    _finite_float(prev_row.get("flow_backward_consistency", prev_row.get("flow_reliable_ratio", 0.0)), 0.0),
                    _finite_float(next_row.get("flow_backward_consistency", next_row.get("flow_reliable_ratio", 0.0)), 0.0),
                )
                work.at[run_idx, "flow_measurement_confidence"] = min(
                    0.25,
                    max(
                        0.0,
                        0.5 * min(
                            _finite_float(prev_row.get("flow_measurement_confidence", 0.0), 0.0),
                            _finite_float(next_row.get("flow_measurement_confidence", 0.0), 0.0),
                        ),
                    ),
                )
                bridge_indices.append(run_idx)
    if not bridge_indices:
        return grid_rows
    return work.sort_values(["sample_index", "grid_y", "grid_x"]).to_dict("records")


def compute_event_grid_flow_from_frame_iter(
    frame_iter: Iterable[Any],
    *,
    flow_sample_sec: float,
    flow_analysis_scale: float,
    flow_grid: tuple[int, int],
    reliable_error_threshold_px: float,
    motion_gap_interpolation_max_sec: float = DEFAULT_FLOW_GAP_INTERPOLATION_MAX_SEC,
    motion_gap_min_vector_mag: float = DEFAULT_FLOW_GAP_MIN_VECTOR_MAG,
    motion_gap_direction_cosine: float = DEFAULT_FLOW_GAP_DIRECTION_COSINE,
) -> pd.DataFrame:
    sample_sec = max(float(flow_sample_sec), 1e-6)
    iterator = iter(frame_iter)
    try:
        first_item = next(iterator)
    except StopIteration:
        return pd.DataFrame()
    first_frame_index, first_time_sec, first_frame, first_metadata = _frame_item_parts(first_item)

    prev_gray, _, _ = resize_gray_for_flow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY), flow_analysis_scale)
    prev_time = float(first_time_sec)
    prev_frame_index = int(first_frame_index)
    frame_shape = prev_gray.shape[:2]
    accumulators: dict[tuple[int, int, int], dict[str, Any]] = {}
    previous_motion_summary: dict[str, float] | None = None
    duration_candidates = [_finite_float(first_metadata.get("source_duration_sec", np.nan), np.nan)]

    for item in iterator:
        frame_index, current_time_sec, frame, metadata = _frame_item_parts(item)
        current_time = float(current_time_sec)
        source_start = prev_time
        source_end = current_time
        if not np.isfinite(source_end) or source_end <= source_start:
            source_end = source_start + sample_sec
        source_dt = max(source_end - source_start, 1e-12)
        gap_capture_frames = _finite_float(metadata.get("gap_capture_frames", frame_index - prev_frame_index), np.nan)
        duration_candidates.append(_finite_float(metadata.get("source_duration_sec", np.nan), np.nan))
        try:
            gray, scale_x, scale_y = resize_gray_for_flow(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), flow_analysis_scale)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **DEFAULT_FARNEBACK_PARAMS)
            backward = cv2.calcOpticalFlowFarneback(gray, prev_gray, None, **DEFAULT_FARNEBACK_PARAMS)
            reliable_mask, forward_backward_error = compute_forward_backward_reliability_and_error(
                flow,
                backward,
                error_threshold_px=reliable_error_threshold_px,
            )
            base_rows = compute_grid_flow_features(
                flow,
                int(frame_index),
                current_time,
                flow_grid,
                sample_index=_event_bin_index(source_end, sample_sec),
                source_dt=source_dt,
                scale_x=scale_x,
                scale_y=scale_y,
                reliable_mask=reliable_mask,
                forward_backward_error=forward_backward_error,
                previous_gray=prev_gray,
                current_gray=gray,
                source_start_sec=source_start,
                source_end_sec=source_end,
                observed_dt_sec=source_dt,
                gap_capture_frames=gap_capture_frames,
                update_frame_index=int(metadata.get("source_frame_index", frame_index)),
            )
        except Exception:
            gray, _, _ = resize_gray_for_flow(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), flow_analysis_scale)
            base_rows = zero_grid_flow_rows(
                gray.shape[:2],
                flow_grid,
                frame_index=int(frame_index),
                time_sec=current_time,
                sample_index=_event_bin_index(source_end, sample_sec),
                reliable_ratio=0.0,
                flow_mode="flow_failed",
                flow_confidence=0.0,
                observed_dt_sec=source_dt,
                source_start_sec=source_start,
                source_end_sec=source_end,
                gap_capture_frames=gap_capture_frames,
                flow_hold=0.0,
                update_frame_index=int(metadata.get("source_frame_index", frame_index)),
            )
            for row in base_rows:
                row["source_dt"] = source_dt

        current_motion_summary = _grid_flow_motion_summary(
            base_rows,
            active_mag_threshold=motion_gap_min_vector_mag,
        )
        motion_gap_active, direction_aligned = _should_interpolate_motion_gap(
            previous_motion_summary,
            current_motion_summary,
            source_dt=source_dt,
            max_gap_sec=motion_gap_interpolation_max_sec,
            min_vector_mag=motion_gap_min_vector_mag,
            min_active_cell_ratio=DEFAULT_FLOW_GAP_MIN_ACTIVE_CELL_RATIO,
            direction_cosine_threshold=motion_gap_direction_cosine,
        )
        if source_dt <= sample_sec * 1.5:
            for row in base_rows:
                row.update({
                    "flow_mode": "observed_update",
                    "flow_confidence": 1.0,
                    "flow_interpolated": 0.0,
                    "flow_hold": 0.0,
                })
            accumulate_grid_flow_rows(
                accumulators,
                base_rows,
                start_sec=source_start,
                end_sec=source_end,
                bin_sec=sample_sec,
            )
        elif motion_gap_active and direction_aligned:
            for row in base_rows:
                row.update({
                    "flow_mode": "interpolated_motion_gap",
                    "flow_confidence": max(0.3, min(1.0, sample_sec / source_dt)),
                    "flow_interpolated": 1.0,
                    "flow_hold": 0.0,
                })
            accumulate_grid_flow_rows(
                accumulators,
                base_rows,
                start_sec=source_start,
                end_sec=source_end,
                bin_sec=sample_sec,
            )
        elif motion_gap_active:
            for row in base_rows:
                row.update({
                    "flow_mode": "uncertain_motion_gap_apportioned",
                    "flow_confidence": max(0.15, min(0.5, sample_sec / source_dt)),
                    "flow_interpolated": 1.0,
                    "flow_hold": 0.0,
                })
            accumulate_grid_flow_rows(
                accumulators,
                base_rows,
                start_sec=source_start,
                end_sec=source_end,
                bin_sec=sample_sec,
            )
        else:
            target_bin = _event_bin_index(source_end, sample_sec)
            for row in base_rows:
                row.update({
                    "flow_mode": "uncertain_gap",
                    "flow_confidence": max(0.2, min(0.8, sample_sec / source_dt)),
                    "flow_interpolated": 0.0,
                    "flow_hold": 0.0,
                })
            accumulate_grid_flow_rows_to_bin(accumulators, base_rows, bin_index=target_bin, bin_sec=sample_sec)

        previous_motion_summary = current_motion_summary
        prev_gray = gray
        prev_time = source_end
        prev_frame_index = int(frame_index)

    finite_durations = [value for value in duration_candidates if np.isfinite(value) and value > 0.0]
    duration_sec = max(finite_durations) if finite_durations else max(prev_time + sample_sec, sample_sec)
    complete_grid_flow_bins(
        accumulators,
        frame_shape,
        flow_grid,
        duration_sec=duration_sec,
        bin_sec=sample_sec,
    )
    grid_rows = finalize_grid_flow_window_rows(accumulators, bin_sec=sample_sec)
    grid_rows = bridge_isolated_motion_gap_rows(
        grid_rows,
        min_vector_mag=motion_gap_min_vector_mag,
    )
    if not grid_rows:
        grid_rows = zero_grid_flow_rows(
            frame_shape,
            flow_grid,
            frame_index=int(first_frame_index),
            time_sec=float(first_time_sec),
            sample_index=0,
            reliable_ratio=0.0,
        )
        for row in grid_rows:
            row["valid_ratio"] = 0.0
    return pd.DataFrame(grid_rows).sort_values(["sample_index", "grid_y", "grid_x"]).reset_index(drop=True)


def compute_raw_flow_from_frame_iter(
    frame_iter: Iterable[Any],
    prefix: str,
    *,
    video_id: str,
    flow_sample_sec: float,
    flow_analysis_scale: float,
    flow_grid: tuple[int, int],
    reliable_error_threshold_px: float,
    event_based: bool = False,
) -> pd.DataFrame:
    sample_sec = max(float(flow_sample_sec), 1e-6)
    if event_based:
        grid_flow_df = compute_event_grid_flow_from_frame_iter(
            frame_iter,
            flow_sample_sec=sample_sec,
            flow_analysis_scale=flow_analysis_scale,
            flow_grid=flow_grid,
            reliable_error_threshold_px=reliable_error_threshold_px,
        )
    else:
        grid_flow_df = compute_grid_flow_from_frame_iter(
            frame_iter,
            flow_sample_sec=sample_sec,
            flow_analysis_scale=flow_analysis_scale,
            flow_grid=flow_grid,
            reliable_error_threshold_px=reliable_error_threshold_px,
        )
    return feature_cache.visualization_grid_to_raw_flow(
        grid_flow_df,
        sample_id=video_id,
        view=prefix,
        flow_sample_sec=sample_sec,
    )


def extract_video_grid_flow(
    video_path: str | Path,
    *,
    flow_sample_sec: float,
    frame_resize_width: int | None,
    flow_analysis_scale: float,
    flow_grid: tuple[int, int],
    reliable_error_threshold_px: float,
    frame_map_path: str | Path | None = None,
    static_frame_max_grid_vector_mag: float | None = None,
) -> pd.DataFrame:
    frame_map = load_frame_time_map(frame_map_path)
    use_static_frame_split = (
        static_frame_max_grid_vector_mag is not None
        and np.isfinite(static_frame_max_grid_vector_mag)
        and float(static_frame_max_grid_vector_mag) >= 0.0
    )
    if use_static_frame_split:
        frame_iter = iter_video_frames_with_time_map(
            video_path,
            resize_width=frame_resize_width,
            frame_map_path=frame_map_path,
        ) if not frame_map.empty else iter_video_frames(video_path, resize_width=frame_resize_width)
        return compute_static_frame_split_grid_flow_from_frame_iter(
            frame_iter,
            flow_sample_sec=flow_sample_sec,
            flow_analysis_scale=flow_analysis_scale,
            flow_grid=flow_grid,
            reliable_error_threshold_px=reliable_error_threshold_px,
            static_frame_max_grid_vector_mag=float(static_frame_max_grid_vector_mag),
        )
    if not frame_map.empty:
        return compute_event_grid_flow_from_frame_iter(
            iter_video_frames_with_time_map(video_path, resize_width=frame_resize_width, frame_map_path=frame_map_path),
            flow_sample_sec=flow_sample_sec,
            flow_analysis_scale=flow_analysis_scale,
            flow_grid=flow_grid,
            reliable_error_threshold_px=reliable_error_threshold_px,
        )
    return compute_grid_flow_from_frame_iter(
        iter_video_frames(video_path, resize_width=frame_resize_width),
        flow_sample_sec=flow_sample_sec,
        flow_analysis_scale=flow_analysis_scale,
        flow_grid=flow_grid,
        reliable_error_threshold_px=reliable_error_threshold_px,
    )


def _empty_flow_row(prefix: str, video_id: str, time_sec: float, time_bin: int, grid: tuple[int, int]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "video_id": video_id,
        "time": float(time_sec),
        "time_bin": int(time_bin),
        f"{prefix}_flow_failed": 1.0,
        f"{prefix}_flow_reliable_ratio": 0.0,
        f"{prefix}_flow_backward_consistency": 0.0,
        f"{prefix}_flow_edge_confidence": 0.0,
        f"{prefix}_flow_coherence_confidence": 1.0,
        f"{prefix}_flow_measurement_confidence": 0.0,
    }
    for gy in range(int(grid[0])):
        for gx in range(int(grid[1])):
            for name in [
                "mag_mean",
                "mag_std",
                "x_mean",
                "y_mean",
                "valid_ratio",
                "flow_reliable_ratio",
                "flow_backward_consistency",
                "flow_edge_strength",
                "flow_edge_density",
                "flow_edge_confidence",
                "flow_coherence_confidence",
                "flow_measurement_confidence",
            ]:
                row[f"{prefix}_flow_cell_{gy}_{gx}_{name}"] = 0.0
    return row


def compute_optical_flow_features(
    frames: list[dict[str, Any]],
    prefix: str,
    *,
    video_id: str,
    flow_sample_sec: float,
    flow_analysis_scale: float,
    flow_grid: tuple[int, int],
    reliable_error_threshold_px: float,
) -> pd.DataFrame:
    return compute_raw_flow_from_frame_iter(
        ((idx, float(item.get("time", 0.0)), item["frame"]) for idx, item in enumerate(frames)),
        prefix,
        video_id=video_id,
        flow_sample_sec=flow_sample_sec,
        flow_analysis_scale=flow_analysis_scale,
        flow_grid=flow_grid,
        reliable_error_threshold_px=reliable_error_threshold_px,
    )


def extract_video_raw_flow(
    video_path: str | Path,
    prefix: str,
    *,
    video_id: str,
    flow_sample_sec: float,
    frame_resize_width: int | None,
    flow_analysis_scale: float,
    flow_grid: tuple[int, int],
    reliable_error_threshold_px: float,
    frame_map_path: str | Path | None = None,
) -> pd.DataFrame:
    frame_map = load_frame_time_map(frame_map_path)
    if not frame_map.empty:
        return compute_raw_flow_from_frame_iter(
            iter_video_frames_with_time_map(video_path, resize_width=frame_resize_width, frame_map_path=frame_map_path),
            prefix,
            video_id=video_id,
            flow_sample_sec=flow_sample_sec,
            flow_analysis_scale=flow_analysis_scale,
            flow_grid=flow_grid,
            reliable_error_threshold_px=reliable_error_threshold_px,
            event_based=True,
        )
    return compute_raw_flow_from_frame_iter(
        iter_video_frames(video_path, resize_width=frame_resize_width),
        prefix,
        video_id=video_id,
        flow_sample_sec=flow_sample_sec,
        flow_analysis_scale=flow_analysis_scale,
        flow_grid=flow_grid,
        reliable_error_threshold_px=reliable_error_threshold_px,
    )


def extract_sample_raw_flow(
    sample: pd.Series | dict[str, Any],
    *,
    use_front: bool,
    use_rear: bool,
    flow_sample_sec: float,
    frame_resize_width: int | None,
    flow_analysis_scale: float,
    flow_grid: tuple[int, int],
    reliable_error_threshold_px: float,
) -> pd.DataFrame:
    sample_series = pd.Series(sample)
    sample_id = str(sample_series.get("sample_id", "unknown"))
    parts: list[pd.DataFrame] = []
    for prefix, enabled, path_col in [("front", use_front, "front_path"), ("rear", use_rear, "rear_path")]:
        path_value = sample_series.get(path_col)
        if not enabled or path_value is None or pd.isna(path_value) or not Path(path_value).exists():
            continue
        frame_map_path = sample_series.get(f"{prefix}_frame_map_path")
        parts.append(extract_video_raw_flow(
            path_value,
            prefix,
            video_id=sample_id,
            flow_sample_sec=flow_sample_sec,
            frame_resize_width=frame_resize_width,
            flow_analysis_scale=flow_analysis_scale,
            flow_grid=flow_grid,
            reliable_error_threshold_px=reliable_error_threshold_px,
            frame_map_path=frame_map_path,
        ))
    return feature_cache.merge_raw_flow_dfs(parts, time_bin_sec=flow_sample_sec)
