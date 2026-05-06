from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def split_group_labels(group_count: int) -> list[str]:
    """Return stable display labels for split-validation groups."""
    count = int(group_count)
    if count < 2:
        raise ValueError("group_count must be at least 2")
    if count <= 26:
        return [chr(ord("A") + index) for index in range(count)]
    return [f"G{index + 1:02d}" for index in range(count)]


def assign_random_split_groups(
    samples_df: pd.DataFrame,
    *,
    group_count: int,
    random_state: int,
) -> pd.DataFrame:
    """Assign samples to nearly-even random groups while preserving row order."""
    if samples_df.empty:
        raise ValueError("samples_df must contain at least one sample")
    labels = split_group_labels(group_count)
    if len(samples_df) < len(labels):
        raise ValueError(f"group_count={group_count} exceeds sample count={len(samples_df)}")

    out = samples_df.reset_index(drop=True).copy()
    positions = np.arange(len(out))
    rng = np.random.default_rng(int(random_state))
    shuffled_positions = rng.permutation(positions)
    chunks = np.array_split(shuffled_positions, len(labels))

    assigned_label: dict[int, str] = {}
    assigned_index: dict[int, int] = {}
    for group_index, (label, chunk) in enumerate(zip(labels, chunks, strict=True)):
        for position in chunk.tolist():
            assigned_label[int(position)] = label
            assigned_index[int(position)] = group_index

    out["split_group"] = [assigned_label[int(position)] for position in positions]
    out["split_group_index"] = [assigned_index[int(position)] for position in positions]
    return out


def sample_list_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "sample_id",
        "category",
        "environment",
        "plot_label",
        "split_group",
        "split_group_index",
        "front_path",
        "rear_path",
        "audio_path",
    ]
    return [column for column in preferred if column in df.columns]


def write_sample_list(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[sample_list_columns(df)].to_csv(path, index=False)
    return path


def ordered_group_pairs(labels: Iterable[str]) -> list[tuple[str, str]]:
    ordered = list(labels)
    return [(train_group, target_group) for train_group in ordered for target_group in ordered if target_group != train_group]


def dataframe_to_markdown(
    df: pd.DataFrame,
    *,
    columns: Iterable[str] | None = None,
    max_rows: int | None = 20,
) -> str:
    """Render a small DataFrame as a Markdown table without optional tabulate."""
    if df is None or df.empty:
        return "_No rows._"
    table = df.copy()
    if columns is not None:
        selected_columns = [column for column in columns if column in table.columns]
        table = table[selected_columns]
    if max_rows is not None and int(max_rows) >= 0:
        table = table.head(int(max_rows))
    if table.empty or not len(table.columns):
        return "_No rows._"

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    headers = [str(column) for column in table.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(fmt(row[column]) for column in table.columns) + " |")
    return "\n".join(lines)


def binary_score_metrics(labels: pd.Series, scores: pd.Series) -> dict[str, float]:
    y_true = pd.to_numeric(labels, errors="coerce").fillna(0).astype(int)
    y_score = pd.to_numeric(scores, errors="coerce")
    valid = y_true.notna() & y_score.notna()
    y_true = y_true[valid]
    y_score = y_score[valid]
    if y_true.nunique(dropna=True) < 2:
        return {"roc_auc": float("nan"), "average_precision": float("nan")}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        return {
            "roc_auc": float(roc_auc_score(y_true, y_score)),
            "average_precision": float(average_precision_score(y_true, y_score)),
        }
    except Exception:
        return {"roc_auc": float("nan"), "average_precision": float("nan")}


def write_inference_markdown_report(
    *,
    output_path: str | Path,
    pair_label: str,
    train_group: str,
    target_group: str,
    model_path: str | Path,
    normal_count: int,
    anomaly_count: int,
    inference_result_table: pd.DataFrame,
    video_summary: pd.DataFrame,
    top_windows: pd.DataFrame,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_text = ""
    if video_summary is not None and not video_summary.empty and "target_category" in video_summary.columns and "max_anomaly_score" in video_summary.columns:
        y_true = video_summary["target_category"].astype(str).eq("anomaly").astype(int)
        metrics = binary_score_metrics(y_true, video_summary["max_anomaly_score"])
        metrics_df = pd.DataFrame([{**metrics, "n_samples": len(video_summary), "n_anomaly": int(y_true.sum()), "n_normal": int((1 - y_true).sum())}])
        metrics_text = dataframe_to_markdown(metrics_df, max_rows=1)
    else:
        metrics_text = "_No metrics available._"

    all_video_score_columns = [
        "ファイル名",
        "正解ラベル",
        "環境",
        "推論データ群",
        "最大異常窓開始時刻",
        "異常スコア",
        "同期異常スコア",
        "音声異常スコア",
        "動き異常スコア",
    ]
    all_video_score_table = inference_result_table.copy()
    if "異常スコア" in all_video_score_table.columns:
        all_video_score_table = all_video_score_table.sort_values("異常スコア", ascending=False).reset_index(drop=True)
    result_columns = [
        *all_video_score_columns,
        "動き異常カメラ",
        "動き異常成分",
        "動き異常グリッド",
    ]
    summary_columns = [
        "video_id",
        "target_category",
        "target_environment",
        "target_group",
        "max_anomaly_window_start_sec",
        "max_anomaly_score",
        "sync_score_at_max_anomaly",
        "audio_anomaly_score_at_max_anomaly",
        "motion_anomaly_score_at_max_anomaly",
        "top5_mean_anomaly_score",
        "p95_anomaly_score",
        "n_windows",
    ]
    top_window_columns = [
        "video_id",
        "target_category",
        "target_environment",
        "target_group",
        "window_start_sec",
        "time",
        "anomaly_score",
        "audio_anomaly_score",
        "motion_anomaly_score",
        "sync_score",
        "motion_top_camera",
        "motion_top_grid_channel",
        "motion_top_grid_label",
    ]

    lines = [
        f"# 推論結果 {pair_label}",
        "",
        f"- 学習モデル群: {train_group}",
        f"- 推論平常データ群: {target_group}",
        f"- 平常データ件数: {int(normal_count)}",
        f"- 異常データ件数: {int(anomaly_count)}",
        f"- model artifact: `{Path(model_path)}`",
        "",
        "## 簡易評価",
        "",
        metrics_text,
        "",
        "## 全動画スコア",
        "",
        "各スコアは、動画内で最も異常スコアが高かった窓の値です。異常スコアの降順で表示します。",
        "",
        dataframe_to_markdown(all_video_score_table, columns=all_video_score_columns, max_rows=None),
        "",
        "## 動画単位スコア",
        "",
        dataframe_to_markdown(video_summary, columns=summary_columns, max_rows=None),
        "",
        "## 推論結果サマリ",
        "",
        dataframe_to_markdown(inference_result_table, columns=result_columns, max_rows=None),
        "",
        "## 上位異常窓",
        "",
        dataframe_to_markdown(top_windows, columns=top_window_columns, max_rows=30),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
