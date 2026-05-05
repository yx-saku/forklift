from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_CONTENT_UPDATE_DIFF_THRESHOLD = 1.0
DEFAULT_CONTENT_UPDATE_RESIZE_WIDTH = 480
DEFAULT_CONTENT_UPDATE_GAUSSIAN_BLUR_KERNEL: int | None = None
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


def is_content_update(diff_score: float, *, diff_threshold: float = DEFAULT_CONTENT_UPDATE_DIFF_THRESHOLD) -> bool:
    """Return True when a frame differs enough from the previous frame to keep."""
    return bool(np.isfinite(diff_score) and float(diff_score) >= float(diff_threshold))


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
