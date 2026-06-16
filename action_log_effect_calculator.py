import re
import unicodedata

import numpy as np
import pandas as pd

from action_log_effect_config import (
    EFFECT_MONTH_OFFSETS,
    EFFECT_TYPE_LABELS,
    EXPECTED_SUBJECTS_COUNT,
    FAST_EFFECT_SUBJECTS,
    LONG_EFFECT_SUBJECTS,
    MEDIUM_EFFECT_SUBJECTS,
    SUBJECT_EFFECT_MAP,
)


METRIC_COLUMNS = ["gbb", "roomnights", "sales_volumes_rub", "revenue_rub"]
EXCLUSION_PRIORITY = [
    "missing_hotel_id",
    "missing_subject",
    "unknown_subject",
    "invalid_action_date",
    "missing_before_month",
    "missing_after_month",
    "missing_before_and_after_month",
    "missing_metric_before",
    "missing_metric_after",
]


def normalize_subject(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.strip().casefold()
    text = text.replace("’", "'")
    for dash in ("‐", "‑", "‒", "–", "—", "−"):
        text = text.replace(dash, "-")
    text = re.sub(r"\s+", " ", text)
    return text


NORMALIZED_SUBJECT_EFFECT_MAP = {
    normalize_subject(subject): effect_type
    for subject, effect_type in SUBJECT_EFFECT_MAP.items()
}


def validate_effect_config():
    groups = {
        "fast": FAST_EFFECT_SUBJECTS,
        "medium": MEDIUM_EFFECT_SUBJECTS,
        "long": LONG_EFFECT_SUBJECTS,
    }
    seen = {}
    normalized_seen = {}
    errors = []
    for group_name, subjects in groups.items():
        for subject in subjects:
            normalized_subject = normalize_subject(subject)
            if not normalized_subject:
                errors.append(f"в группе {group_name} есть пустой subject")
                continue
            clean_subject = str(subject).strip()
            if clean_subject in seen:
                errors.append(
                    f'subject "{clean_subject}" указан более чем в одной группе эффекта: '
                    f"{seen[clean_subject]} и {group_name}"
                )
            if normalized_subject in normalized_seen:
                previous_subject, previous_group = normalized_seen[normalized_subject]
                errors.append(
                    f'subject "{clean_subject}" после нормализации совпадает с "{previous_subject}": '
                    f"{previous_group} и {group_name}"
                )
            seen[clean_subject] = group_name
            normalized_seen[normalized_subject] = (clean_subject, group_name)
    if len(seen) != EXPECTED_SUBJECTS_COUNT:
        errors.append(
            f"справочник содержит {len(seen)} уникальных subject вместо {EXPECTED_SUBJECTS_COUNT}"
        )
    if set(SUBJECT_EFFECT_MAP.keys()) != set(seen.keys()):
        errors.append("SUBJECT_EFFECT_MAP не совпадает с наборами subject")
    if len(NORMALIZED_SUBJECT_EFFECT_MAP) != EXPECTED_SUBJECTS_COUNT:
        errors.append(
            f"нормализованный справочник содержит {len(NORMALIZED_SUBJECT_EFFECT_MAP)} ключей вместо {EXPECTED_SUBJECTS_COUNT}"
        )
    if errors:
        raise ValueError("Ошибка конфигурации Action Logs:\n" + "\n".join(errors))
    return True


def get_effective_month(action_date):
    action_date = pd.Timestamp(action_date)
    month = action_date.to_period("M")
    if action_date.day <= 15:
        return month
    return month + 1


def get_analysis_months(effective_month, effect_type):
    if effect_type not in EFFECT_MONTH_OFFSETS or pd.isna(effective_month):
        return None, None
    before_offset, after_offset = EFFECT_MONTH_OFFSETS[effect_type]
    return effective_month + before_offset, effective_month + after_offset


def assign_effect_type(subject):
    normalized_subject = normalize_subject(subject)
    if not normalized_subject:
        return "unknown"
    return NORMALIZED_SUBJECT_EFFECT_MAP.get(normalized_subject, "unknown")


def _to_period_month(series):
    if pd.api.types.is_period_dtype(series):
        return series.astype("period[M]")
    return pd.to_datetime(series, errors="coerce").dt.to_period("M")


def _sum_preserve_missing(values):
    return values.sum(min_count=1)


def prepare_monthly_metrics(hotel_data):
    if hotel_data is None or hotel_data.empty:
        return pd.DataFrame(columns=["hotel_id", "calendar_month", *METRIC_COLUMNS, "has_metric_observation"])
    df = hotel_data.copy()
    df["hotel_id"] = df["hotel_id"].astype("string").str.strip()
    if "calendar_month" not in df.columns:
        if "month" in df.columns:
            df["calendar_month"] = _to_period_month(df["month"])
        elif "booking_created_date" in df.columns:
            df["calendar_month"] = _to_period_month(df["booking_created_date"])
        else:
            raise ValueError("monthly_metrics должен содержать month, calendar_month или booking_created_date")
    else:
        df["calendar_month"] = _to_period_month(df["calendar_month"])
    df = df.dropna(subset=["hotel_id", "calendar_month"])
    for metric_column in METRIC_COLUMNS:
        if metric_column not in df.columns:
            df[metric_column] = np.nan
        df[metric_column] = pd.to_numeric(df[metric_column], errors="coerce")
    if "has_source_row" in df.columns:
        df["has_metric_observation"] = df["has_source_row"].fillna(False).astype(bool)
    else:
        df["has_metric_observation"] = True
    monthly_metrics = (
        df.groupby(["hotel_id", "calendar_month"], as_index=False)
        .agg(
            gbb=("gbb", _sum_preserve_missing),
            roomnights=("roomnights", _sum_preserve_missing),
            sales_volumes_rub=("sales_volumes_rub", _sum_preserve_missing),
            revenue_rub=("revenue_rub", _sum_preserve_missing),
            has_metric_observation=("has_metric_observation", "max"),
        )
    )
    monthly_metrics["hotel_id"] = monthly_metrics["hotel_id"].astype("string").str.strip()
    monthly_metrics["calendar_month"] = monthly_metrics["calendar_month"].astype("period[M]")
    return monthly_metrics


def _status_from_change(absolute_change):
    if pd.isna(absolute_change):
        return ""
    if absolute_change > 0:
        return "growth"
    if absolute_change < 0:
        return "decline"
    return "no_change"


def _status_label(change_direction):
    return {
        "growth": "Рост",
        "decline": "Снижение",
        "no_change": "Без изменений",
    }.get(change_direction, "")


def _is_blank(value):
    if value is None or pd.isna(value):
        return True
    return str(value).strip().casefold() in {"", "nan", "none", "nat", "<na>"}


def _first_exclusion_reason(row):
    if _is_blank(row.get("hotel_id_clean")):
        return "missing_hotel_id"
    if _is_blank(row.get("subject_clean")):
        return "missing_subject"
    if row.get("effect_type") == "unknown":
        return "unknown_subject"
    if pd.isna(row.get("action_date")):
        return "invalid_action_date"
    before_missing = pd.isna(row.get("before_month"))
    after_missing = pd.isna(row.get("after_month"))
    if before_missing and after_missing:
        return "missing_before_and_after_month"
    if before_missing:
        return "missing_before_month"
    if after_missing:
        return "missing_after_month"
    metric_before_missing = pd.isna(row.get("metric_before"))
    metric_after_missing = pd.isna(row.get("metric_after"))
    if metric_before_missing and metric_after_missing:
        return "missing_before_and_after_month"
    if metric_before_missing:
        return "missing_metric_before"
    if metric_after_missing:
        return "missing_metric_after"
    return None


def calculate_action_log_effects(action_logs: pd.DataFrame, monthly_metrics: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    validate_effect_config()
    if metric_column not in METRIC_COLUMNS:
        raise ValueError(f"Неизвестная метрика для Action Logs: {metric_column}")

    actions = action_logs.copy() if action_logs is not None else pd.DataFrame()
    if actions.empty:
        return pd.DataFrame(columns=[
            "action_id", "hotel_id", "action_date", "subject", "subject_normalized", "outcome",
            "effect_type", "effect_type_label", "effective_month", "before_month", "after_month",
            "metric_name", "metric_before", "metric_after", "absolute_change", "percentage_change",
            "change_direction", "change_direction_label", "zero_before_value",
            "calculation_status", "exclusion_reason",
        ])

    actions = actions.reset_index(drop=True)
    if "action_id" not in actions.columns:
        actions["action_id"] = np.arange(1, len(actions) + 1)
    for column in ["hotel_id", "subject", "outcome", "action_date"]:
        if column not in actions.columns:
            actions[column] = pd.NA

    actions["hotel_id_clean"] = actions["hotel_id"].astype("string").str.strip()
    actions["hotel_id_clean"] = actions["hotel_id_clean"].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NaT": pd.NA})
    actions["subject_clean"] = actions["subject"].astype("string").str.strip()
    actions["subject_clean"] = actions["subject_clean"].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NaT": pd.NA})
    actions["subject_normalized"] = actions["subject"].map(normalize_subject)
    actions["effect_type"] = actions["subject_normalized"].map(NORMALIZED_SUBJECT_EFFECT_MAP).fillna("unknown")
    actions["effect_type_label"] = actions["effect_type"].map(EFFECT_TYPE_LABELS).fillna(EFFECT_TYPE_LABELS["unknown"])
    actions["action_date"] = pd.to_datetime(actions["action_date"], errors="coerce")

    valid_dates = actions["action_date"].notna()
    empty_period_months = pd.Series(pd.NaT, index=actions.index).dt.to_period("M")
    actions["effective_month"] = empty_period_months.copy()
    actions.loc[valid_dates, "effective_month"] = actions.loc[valid_dates, "action_date"].map(get_effective_month)
    actions["before_month"] = empty_period_months.copy()
    actions["after_month"] = empty_period_months.copy()
    known_effect = actions["effect_type"].ne("unknown") & actions["effective_month"].notna()
    for index, row in actions.loc[known_effect, ["effective_month", "effect_type"]].iterrows():
        before_month, after_month = get_analysis_months(row["effective_month"], row["effect_type"])
        actions.at[index, "before_month"] = before_month
        actions.at[index, "after_month"] = after_month

    actions["hotel_id"] = actions["hotel_id_clean"].astype("string")
    actions["effective_month"] = actions["effective_month"].astype("period[M]")
    actions["before_month"] = actions["before_month"].astype("period[M]")
    actions["after_month"] = actions["after_month"].astype("period[M]")

    metrics = prepare_monthly_metrics(monthly_metrics)
    before_metrics = metrics.rename(
        columns={"calendar_month": "before_month", metric_column: "metric_before"}
    )[["hotel_id", "before_month", "metric_before", "has_metric_observation"]].rename(
        columns={"has_metric_observation": "has_metric_before_observation"}
    )
    after_metrics = metrics.rename(
        columns={"calendar_month": "after_month", metric_column: "metric_after"}
    )[["hotel_id", "after_month", "metric_after", "has_metric_observation"]].rename(
        columns={"has_metric_observation": "has_metric_after_observation"}
    )

    result = actions.merge(before_metrics, on=["hotel_id", "before_month"], how="left")
    result = result.merge(after_metrics, on=["hotel_id", "after_month"], how="left")
    result.loc[result["has_metric_before_observation"].eq(False), "metric_before"] = np.nan
    result.loc[result["has_metric_after_observation"].eq(False), "metric_after"] = np.nan
    result["metric_name"] = metric_column
    result["exclusion_reason"] = result.apply(_first_exclusion_reason, axis=1)
    result["calculation_status"] = np.where(result["exclusion_reason"].isna(), "calculated", "excluded")
    result["absolute_change"] = result["metric_after"] - result["metric_before"]
    result["zero_before_value"] = result["metric_before"].eq(0) & result["metric_before"].notna()
    result["percentage_change"] = np.where(
        result["zero_before_value"],
        np.nan,
        result["absolute_change"] / result["metric_before"] * 100,
    )
    result.loc[result["calculation_status"].eq("excluded"), ["absolute_change", "percentage_change"]] = np.nan
    result["change_direction"] = result["absolute_change"].apply(_status_from_change)
    result["change_direction_label"] = result["change_direction"].apply(_status_label)
    result["outcome"] = result["outcome"].astype("string").str.strip()
    result["subject"] = result["subject_clean"].astype("string")

    output_columns = [
        "action_id", "hotel_id", "action_date", "subject", "subject_normalized", "outcome",
        "effect_type", "effect_type_label", "effective_month", "before_month", "after_month",
        "metric_name", "metric_before", "metric_after", "absolute_change", "percentage_change",
        "change_direction", "change_direction_label", "zero_before_value",
        "calculation_status", "exclusion_reason",
    ]
    return result[output_columns].replace({pd.NA: np.nan})
