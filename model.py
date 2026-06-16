import numpy as np
import pandas as pd

from config import METRICS, MONTH_NAMES
from utils import safe_divide


def calculate_seasonality_by_segment(monthly_df, metric_name):
    segment_totals = (
        monthly_df.groupby(["segment_key", "month", "month_num"], as_index=False)
        [metric_name].sum()
    )
    avg_by_month = (
        segment_totals.groupby(["segment_key", "month_num"], as_index=False)
        [metric_name].mean()
    )
    overall_avg = (
        segment_totals.groupby("segment_key", as_index=False)
        [metric_name].mean()
        .rename(columns={metric_name: "overall_avg"})
    )
    seasonality = avg_by_month.merge(overall_avg, on="segment_key", how="left")
    seasonality["seasonality_index"] = seasonality.apply(
        lambda row: safe_divide(row[metric_name], row["overall_avg"]), axis=1
    )
    seasonality["seasonality_index"] = (
        seasonality["seasonality_index"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1)
    )
    return seasonality[["segment_key", "month_num", "seasonality_index"]]


def calculate_general_seasonality(monthly_df, metric_name):
    totals = (
        monthly_df.groupby(["month", "month_num"], as_index=False)
        [metric_name].sum()
    )
    avg_by_month = (
        totals.groupby("month_num", as_index=False)
        [metric_name].mean()
    )
    overall_avg = totals[metric_name].mean()
    avg_by_month["seasonality_index"] = avg_by_month[metric_name] / overall_avg
    avg_by_month["month_name"] = avg_by_month["month_num"].map(MONTH_NAMES)
    return avg_by_month


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
