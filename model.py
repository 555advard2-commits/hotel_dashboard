import numpy as np
import pandas as pd

from config import METRICS, MONTH_NAMES
from utils import safe_divide


def _weighted_average(values, weights):
    values = pd.Series(values, dtype="float64")
    weights = pd.Series(weights, dtype="float64")
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def _filter_real_observations(monthly_df, metric_name):
    df = monthly_df.copy()
    if "has_source_row" in df.columns:
        df = df[df["has_source_row"].fillna(False).astype(bool)].copy()
    elif "has_metric_observation" in df.columns:
        df = df[df["has_metric_observation"].fillna(False).astype(bool)].copy()
    if df.empty:
        return df
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df.dropna(subset=["hotel_id", "month"])
    df["month_num"] = df["month"].dt.month
    df[metric_name] = pd.to_numeric(df[metric_name], errors="coerce")
    return df.dropna(subset=[metric_name])


def _add_outlier_flag(monthly_avg, group_cols, outlier_threshold):
    monthly_avg = monthly_avg.copy()
    if group_cols:
        group_obj = monthly_avg.groupby(group_cols)["monthly_avg"]
        mean = group_obj.transform("mean")
        std = group_obj.transform("std")
        z_score = (monthly_avg["monthly_avg"] - mean).abs() / std
        monthly_avg["is_outlier"] = z_score.gt(outlier_threshold).fillna(False)
        monthly_avg.loc[std.isna() | std.eq(0), "is_outlier"] = False
    else:
        mean = monthly_avg["monthly_avg"].mean()
        std = monthly_avg["monthly_avg"].std()
        if pd.isna(std) or std == 0:
            monthly_avg["is_outlier"] = False
        else:
            z_score = (monthly_avg["monthly_avg"] - mean).abs() / std
            monthly_avg["is_outlier"] = z_score.gt(outlier_threshold).fillna(False)
    return monthly_avg


def calculate_seasonality_improved(monthly_df, metric_name, group_cols=None, outlier_threshold=3):
    group_cols = group_cols or []
    df = _filter_real_observations(monthly_df, metric_name)
    result_cols = [*group_cols, "month_num", metric_name, "overall_avg", "seasonality_index", "hotel_count"]
    if df.empty:
        return pd.DataFrame(columns=result_cols)

    monthly_avg = (
        df.groupby([*group_cols, "month", "month_num"], as_index=False)
        .agg(
            monthly_avg=(metric_name, "mean"),
            hotel_count=("hotel_id", "nunique"),
        )
    )
    monthly_avg = _add_outlier_flag(monthly_avg, group_cols, outlier_threshold)
    clean = monthly_avg[~monthly_avg["is_outlier"]].copy()
    if clean.empty:
        clean = monthly_avg.copy()

    group_keys = [*group_cols, "month_num"]
    calendar_rows = []
    for key, group in clean.groupby(group_keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {column: value for column, value in zip(group_keys, key)}
        row[metric_name] = _weighted_average(group["monthly_avg"], group["hotel_count"])
        row["hotel_count"] = int(group["hotel_count"].sum())
        calendar_rows.append(row)
    calendar_avg = pd.DataFrame(calendar_rows)

    if group_cols:
        overall_rows = []
        for key, group in clean.groupby(group_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            row = {column: value for column, value in zip(group_cols, key)}
            row["overall_avg"] = _weighted_average(group["monthly_avg"], group["hotel_count"])
            overall_rows.append(row)
        overall_avg = pd.DataFrame(overall_rows)
        seasonality = calendar_avg.merge(overall_avg, on=group_cols, how="left")
    else:
        seasonality = calendar_avg.copy()
        seasonality["overall_avg"] = _weighted_average(clean["monthly_avg"], clean["hotel_count"])

    seasonality["seasonality_index"] = seasonality.apply(
        lambda row: safe_divide(row[metric_name], row["overall_avg"]), axis=1
    )
    seasonality["seasonality_index"] = seasonality["seasonality_index"].replace([np.inf, -np.inf], np.nan).fillna(1)
    return seasonality[result_cols]


def calculate_seasonality_by_segment(monthly_df, metric_name):
    seasonality = calculate_seasonality_improved(monthly_df, metric_name, group_cols=["segment_key"])
    return seasonality[["segment_key", "month_num", "seasonality_index"]]


def calculate_general_seasonality(monthly_df, metric_name):
    seasonality = calculate_seasonality_improved(monthly_df, metric_name)
    seasonality["month_name"] = seasonality["month_num"].map(MONTH_NAMES)
    return seasonality


def add_expected_values(monthly_df, metric_name):
    seasonality = calculate_seasonality_by_segment(monthly_df, metric_name)
    df = monthly_df.merge(
        seasonality, on=["segment_key", "month_num"], how="left"
    )
    df["seasonality_index"] = df["seasonality_index"].fillna(1)
    df["seasonally_adjusted_value"] = df[metric_name] / df["seasonality_index"]
    hotel_baseline = (
        df.groupby("hotel_id", as_index=False)
        ["seasonally_adjusted_value"]
        .mean()
        .rename(columns={"seasonally_adjusted_value": "hotel_baseline"})
    )
    df = df.merge(hotel_baseline, on="hotel_id", how="left")
    df["expected_value"] = df["hotel_baseline"] * df["seasonality_index"]
    df["hotel_efficiency"] = df.apply(
        lambda row: safe_divide(row[metric_name], row["expected_value"]), axis=1
    )
    df["hotel_residual_pct"] = df["hotel_efficiency"] - 1
    segment_month = (
        df.groupby(["segment_key", "month"], as_index=False)
        .agg(
            segment_actual=(metric_name, "sum"),
            segment_expected=("expected_value", "sum")
        )
    )
    segment_month["segment_efficiency"] = segment_month.apply(
        lambda row: safe_divide(row["segment_actual"], row["segment_expected"]), axis=1
    )
    segment_month["segment_residual_pct"] = segment_month["segment_efficiency"] - 1
    df = df.merge(
        segment_month[["segment_key", "month", "segment_efficiency", "segment_residual_pct"]],
        on=["segment_key", "month"],
        how="left"
    )
    df["individual_effect_pct"] = df["hotel_residual_pct"] - df["segment_residual_pct"]
    return df


def calculate_metric_summary(monthly_df):
    rows = []
    for metric_name, metric_label in METRICS.items():
        enriched_metric = add_expected_values(monthly_df, metric_name)
        actual_sum = enriched_metric[metric_name].sum()
        expected_sum = enriched_metric["expected_value"].sum()
        efficiency = safe_divide(actual_sum, expected_sum)
        rows.append({
            "metric": metric_name,
            "metric_label": metric_label,
            "actual_sum": actual_sum,
            "expected_sum": expected_sum,
            "efficiency": efficiency,
            "deviation_pct": efficiency - 1 if not pd.isna(efficiency) else np.nan,
            "description": METRICS[metric_name]
        })
    return pd.DataFrame(rows)


def calculate_hotel_scores(enriched_df, metric_name):
    result = (
        enriched_df.groupby(["hotel_id", "is_STR", "segment_key"], as_index=False)
        .agg(
            actual_sum=(metric_name, "sum"),
            expected_sum=("expected_value", "sum"),
            active_months=(metric_name, lambda x: (x > 0).sum())
        )
    )
    result["efficiency"] = result.apply(
        lambda row: safe_divide(row["actual_sum"], row["expected_sum"]), axis=1
    )
    result["deviation_pct"] = result["efficiency"] - 1
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.dropna(subset=["efficiency"])
    return result
