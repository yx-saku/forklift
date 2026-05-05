from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _to_float_audio(data: np.ndarray) -> np.ndarray:
    values = np.asarray(data)
    if values.ndim > 1:
        values = values.mean(axis=1)
    if np.issubdtype(values.dtype, np.integer):
        max_abs = float(np.iinfo(values.dtype).max)
        values = values.astype(np.float32) / max(max_abs, 1.0)
    else:
        values = values.astype(np.float32)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _read_wav_fallback(audio_path: str | Path) -> tuple[int, np.ndarray]:
    with wave.open(str(audio_path), "rb") as wav:
        sr = int(wav.getframerate())
        channels = int(wav.getnchannels())
        sample_width = int(wav.getsampwidth())
        frames = wav.readframes(wav.getnframes())
    if sample_width == 1:
        dtype = np.uint8
    elif sample_width == 2:
        dtype = np.int16
    elif sample_width == 4:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported wav sample width: {sample_width}")
    data = np.frombuffer(frames, dtype=dtype)
    if channels > 1 and data.size:
        data = data.reshape(-1, channels)
    return sr, data


def _resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if int(orig_sr) == int(target_sr) or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    duration = float(audio.size) / max(float(orig_sr), 1.0)
    target_len = max(1, int(round(duration * float(target_sr))))
    old_x = np.linspace(0.0, duration, num=audio.size, endpoint=False)
    new_x = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(new_x, old_x, audio).astype(np.float32)


def load_audio_mono(audio_path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    path = Path(audio_path)
    if not path.exists():
        return np.zeros((0,), dtype=np.float32), int(target_sr)
    try:
        from scipy.io import wavfile

        sr, data = wavfile.read(path)
    except Exception:
        sr, data = _read_wav_fallback(path)
    audio = _to_float_audio(data)
    target_sr = int(target_sr)
    if int(sr) != target_sr and audio.size:
        try:
            from scipy.signal import resample_poly

            gcd = math.gcd(int(sr), target_sr)
            audio = resample_poly(audio, target_sr // gcd, int(sr) // gcd).astype(np.float32)
        except Exception:
            audio = _resample_linear(audio, int(sr), target_sr)
        sr = target_sr
    return audio.astype(np.float32, copy=False), int(sr)


def _spectral_features(segment: np.ndarray, sr: int, n_bands: int) -> dict[str, float]:
    if segment.size == 0:
        return {
            "audio_centroid": 0.0,
            "audio_bandwidth": 0.0,
            "audio_high_freq_energy": 0.0,
            **{f"audio_mel_{idx:02d}": 0.0 for idx in range(int(n_bands))},
        }
    window = np.hanning(segment.size).astype(np.float32)
    spectrum = np.fft.rfft(segment.astype(np.float32) * window)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(segment.size, d=1.0 / max(int(sr), 1))
    total = float(power.sum())
    if total <= 1e-12:
        centroid = 0.0
        bandwidth = 0.0
        high_freq_energy = 0.0
    else:
        centroid = float((freqs * power).sum() / total)
        bandwidth = float(np.sqrt((((freqs - centroid) ** 2) * power).sum() / total))
        high_freq_energy = float(power[freqs >= 3000.0].sum() / total) if power.size else 0.0

    max_freq = float(freqs[-1]) if freqs.size else float(sr) / 2.0
    edges = np.linspace(0.0, max_freq, int(n_bands) + 1)
    band_values: dict[str, float] = {}
    for idx in range(int(n_bands)):
        mask = (freqs >= edges[idx]) & (freqs < edges[idx + 1])
        band_energy = float(power[mask].mean()) if np.any(mask) else 0.0
        band_values[f"audio_mel_{idx:02d}"] = float(np.log1p(band_energy))
    return {
        "audio_centroid": centroid,
        "audio_bandwidth": bandwidth,
        "audio_high_freq_energy": high_freq_energy,
        **band_values,
    }


def extract_audio_features(
    audio_path: str | Path | None,
    *,
    sample_id: str,
    audio_sr: int = 16000,
    window_sec: float = 0.2,
    hop_sec: float | None = None,
    n_mels: int = 16,
    metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if audio_path is None or pd.isna(audio_path) or not Path(audio_path).exists():
        return pd.DataFrame(columns=["video_id", "sample_id", "time", "time_bin"])
    audio, sr = load_audio_mono(audio_path, target_sr=audio_sr)
    if audio.size == 0:
        return pd.DataFrame(columns=["video_id", "sample_id", "time", "time_bin"])
    window_len = max(1, int(round(float(window_sec) * sr)))
    hop_len = max(1, int(round(float(hop_sec if hop_sec is not None else window_sec) * sr)))
    rows: list[dict[str, Any]] = []
    max_start = max(0, audio.size - window_len)
    starts = list(range(0, max_start + 1, hop_len))
    if not starts or starts[-1] != max_start:
        starts.append(max_start)
    for index, start in enumerate(starts):
        segment = audio[start : start + window_len]
        if segment.size < window_len:
            segment = np.pad(segment, (0, window_len - segment.size))
        abs_segment = np.abs(segment)
        zero_crossings = np.mean(segment[:-1] * segment[1:] < 0.0) if segment.size > 1 else 0.0
        time_sec = float((start + 0.5 * window_len) / max(sr, 1))
        row: dict[str, Any] = {
            "video_id": str(sample_id),
            "sample_id": str(sample_id),
            "time": time_sec,
            "time_bin": int(round(time_sec / max(float(hop_sec if hop_sec is not None else window_sec), 1e-6))),
            "audio_rms": float(np.sqrt(np.mean(segment * segment))),
            "audio_energy": float(np.mean(segment * segment)),
            "audio_peak": float(abs_segment.max()) if abs_segment.size else 0.0,
            "audio_ptp": float(np.ptp(segment)) if segment.size else 0.0,
            "audio_zcr": float(zero_crossings),
            "audio_window_index": int(index),
        }
        row.update(_spectral_features(segment, sr, n_mels))
        if metadata:
            row.update(metadata)
        rows.append(row)
    return pd.DataFrame(rows)
