from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd

from . import cache as feature_cache


DEFAULT_DUPLICATE_FRAME_MEAN_ABS_DIFF_THRESHOLD = 0.5
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
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
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
            "valid_ratio": float(valid_ratio),
            "source_dt": 0.0,
            "flow_window_frame_count": 0.0,
            "flow_forward_backward_error_mean": np.nan,
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
) -> list[dict[str, float]]:
    fx = flow[..., 0] / max(float(scale_x), 1e-6)
    fy = flow[..., 1] / max(float(scale_y), 1e-6)
    mag = np.sqrt(fx * fx + fy * fy)
    rows: list[dict[str, float]] = []
    for cell in grid_bounds(flow.shape[:2], grid):
        x0, y0, x1, y1 = cell["x0"], cell["y0"], cell["x1"], cell["y1"]
        cell_fx = fx[y0:y1, x0:x1]
        cell_fy = fy[y0:y1, x0:x1]
        cell_mag = mag[y0:y1, x0:x1]
        cell_reliable = reliable_mask[y0:y1, x0:x1] if reliable_mask is not None else None
        cell_fb_error = forward_backward_error[y0:y1, x0:x1] if forward_backward_error is not None else None
        mean_dx = float(np.mean(cell_fx)) if cell_fx.size else 0.0
        mean_dy = float(np.mean(cell_fy)) if cell_fy.size else 0.0
        finite_fb_error = cell_fb_error[np.isfinite(cell_fb_error)] if cell_fb_error is not None and cell_fb_error.size else np.asarray([], dtype=float)
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
            "flow_reliable_ratio": float(np.mean(cell_reliable)) if cell_reliable is not None and cell_reliable.size else np.nan,
            "valid_ratio": 1.0,
            "source_dt": float(source_dt) if source_dt is not None else np.nan,
            "flow_window_frame_count": 1.0,
            "flow_forward_backward_error_mean": float(np.mean(finite_fb_error)) if finite_fb_error.size else np.nan,
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
    }


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


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


def finalize_grid_flow_window_rows(accumulators: dict[tuple[int, int, int], dict[str, Any]], *, bin_sec: float) -> list[dict[str, float]]:
    width = max(float(bin_sec), 1e-6)
    rows: list[dict[str, float]] = []
    for _, acc in sorted(accumulators.items(), key=lambda item: item[0]):
        covered_sec = max(float(acc.get("source_dt", 0.0)), 0.0)
        reliable_weight = float(acc.get("flow_reliable_ratio_weight", 0.0))
        error_weight = float(acc.get("flow_forward_backward_error_weight", 0.0))
        mean_dx = float(acc["flow_dx_mean"])
        mean_dy = float(acc["flow_dy_mean"])
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
            "flow_reliable_ratio": float(acc["flow_reliable_ratio_sum"] / reliable_weight) if reliable_weight > 0.0 else 0.0,
            "valid_ratio": float(np.clip(acc["valid_ratio_sum"] / width, 0.0, 1.0)),
            "source_dt": covered_sec,
            "flow_window_frame_count": float(acc["flow_window_frame_count"]),
            "flow_forward_backward_error_mean": float(acc["flow_forward_backward_error_sum"] / error_weight) if error_weight > 0.0 else np.nan,
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


def compute_raw_flow_from_frame_iter(
    frame_iter: Iterable[tuple[int, float, np.ndarray]],
    prefix: str,
    *,
    video_id: str,
    flow_sample_sec: float,
    flow_analysis_scale: float,
    flow_grid: tuple[int, int],
    reliable_error_threshold_px: float,
) -> pd.DataFrame:
    sample_sec = max(float(flow_sample_sec), 1e-6)
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
) -> pd.DataFrame:
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
    }
    for gy in range(int(grid[0])):
        for gx in range(int(grid[1])):
            for name in ["mag_mean", "mag_std", "x_mean", "y_mean", "valid_ratio", "flow_reliable_ratio"]:
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
) -> pd.DataFrame:
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
        parts.append(extract_video_raw_flow(
            path_value,
            prefix,
            video_id=sample_id,
            flow_sample_sec=flow_sample_sec,
            frame_resize_width=frame_resize_width,
            flow_analysis_scale=flow_analysis_scale,
            flow_grid=flow_grid,
            reliable_error_threshold_px=reliable_error_threshold_px,
        ))
    return feature_cache.merge_raw_flow_dfs(parts, time_bin_sec=flow_sample_sec)
