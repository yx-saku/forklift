from __future__ import annotations

from collections.abc import Mapping


DEFAULT_BROAD_VIBRATION_FEATURES: dict[str, dict[str, bool]] = {
    "vector_change": {"score": True, "change_amount": True},
    "flow_reliable_ratio": {"score": False, "change_amount": False},
    "reliability_weighted_vector_change": {"score": False, "change_amount": False},
    "x_change": {"score": False, "change_amount": False},
    "y_change": {"score": False, "change_amount": False},
}

DEFAULT_BROAD_VIBRATION_FEATURE_LABELS: dict[str, str] = {
    "vector_change": "vector change",
    "flow_reliable_ratio": "flow reliable ratio",
    "reliability_weighted_vector_change": "reliability weighted vector change",
    "x_change": "x component change",
    "y_change": "y component change",
}


def build_broad_vibration_column_name(camera: str, feature_name: str) -> str:
    return f"{camera}_broad_vibration_{feature_name}_score"


def build_broad_vibration_change_amount_column_name(camera: str, feature_name: str) -> str:
    return f"{camera}_broad_vibration_{feature_name}_amount_score"


def build_broad_vibration_score_group_name(feature_name: str) -> str:
    return f"broad_vibration_{feature_name}_score"


def build_broad_vibration_change_amount_group_name(feature_name: str) -> str:
    return f"broad_vibration_{feature_name}_amount"


def build_broad_vibration_columns(feature_flags: Mapping[str, object] | None = None) -> list[str]:
    feature_flags = feature_flags or DEFAULT_BROAD_VIBRATION_FEATURES
    columns: list[str] = []
    for feature_name in feature_flags:
        for camera in ["front", "rear"]:
            columns.append(build_broad_vibration_column_name(camera, feature_name))
            columns.append(build_broad_vibration_change_amount_column_name(camera, feature_name))
    return list(dict.fromkeys(columns))


def build_broad_vibration_feature_groups(feature_flags: Mapping[str, Mapping[str, bool]] | None = None) -> dict[str, bool]:
    feature_flags = feature_flags or DEFAULT_BROAD_VIBRATION_FEATURES
    groups: dict[str, bool] = {}
    for feature_name, flags in feature_flags.items():
        groups[build_broad_vibration_score_group_name(feature_name)] = bool(flags.get("score", False))
        groups[build_broad_vibration_change_amount_group_name(feature_name)] = bool(flags.get("change_amount", False))
    return groups


def build_broad_vibration_column_to_group(feature_flags: Mapping[str, object] | None = None) -> dict[str, str]:
    feature_flags = feature_flags or DEFAULT_BROAD_VIBRATION_FEATURES
    column_to_group: dict[str, str] = {}
    for feature_name in feature_flags:
        for camera in ["front", "rear"]:
            column_to_group[build_broad_vibration_column_name(camera, feature_name)] = build_broad_vibration_score_group_name(feature_name)
            column_to_group[build_broad_vibration_change_amount_column_name(camera, feature_name)] = build_broad_vibration_change_amount_group_name(feature_name)
    return column_to_group


def get_broad_vibration_feature_flag(feature_flags: Mapping[str, object], feature_name: str, flag_name: str) -> bool:
    flags = feature_flags.get(feature_name, {})
    return bool(flags.get(flag_name, False)) if isinstance(flags, Mapping) else False
