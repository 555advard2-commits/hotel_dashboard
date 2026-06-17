import csv
import io

import numpy as np
import pandas as pd
import streamlit as st

from config import BOOKING_COLUMNS, ACTION_COLUMNS, METRICS
from utils import (
    parse_bool_series, parse_number_series, mode_value, make_segment_key,
    normalize_columns, classify_outcome_group
)


@st.cache_data(show_spinner=False)
def _read_csv_bytes_cached(raw_bytes):
    encodings = [
        "utf-8-sig", "utf-8", "cp1251",
        "windows-1251", "cp1252", "latin1"
    ]
    last_error = None
    for encoding in encodings:
        try:
            sample = raw_bytes[:65536].decode(encoding, errors="ignore")
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                separator = dialect.delimiter
            except Exception:
                counts = {sep: sample.count(sep) for sep in [",", ";", "\t", "|"]}
                separator = max(counts, key=counts.get)
                if counts[separator] == 0:
                    separator = ","
            return pd.read_csv(
                io.BytesIO(raw_bytes),
                sep=separator,
                engine="c",
                encoding=encoding,
                on_bad_lines="skip",
                low_memory=False,
            )
        except Exception as exc:
            last_error = exc
    try:
        return pd.read_csv(
            io.BytesIO(raw_bytes),
            sep=",",
            engine="c",
            encoding="latin1",
            encoding_errors="replace",
            on_bad_lines="skip",
            low_memory=False,
        )
    except Exception:
        if last_error is not None:
            raise last_error
        return pd.read_csv(io.BytesIO(raw_bytes), sep=None, engine="python", encoding="latin1", encoding_errors="replace", on_bad_lines="skip")


def read_csv_auto(file):
    file.seek(0)
    raw = file.read()
    if isinstance(raw, str):
        raw_bytes = raw.encode("utf-8", errors="replace")
    else:
        raw_bytes = raw
    return _read_csv_bytes_cached(raw_bytes)


def check_columns(df, required_columns, table_name):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.error(f"В таблице {table_name} не хватает колонок: {', '.join(missing)}")
        st.stop()


@st.cache_data
def prepare_bookings_data(bookings_df):
    bookings_df = normalize_columns(bookings_df)
    check_columns(bookings_df, BOOKING_COLUMNS, "продаж")
    df = bookings_df.copy()
    df["hotel_id"] = df["hotel_id"].astype(str).str.strip()
    df["booking_created_date"] = pd.to_datetime(df["booking_created_date"], errors="coerce")
    df = df.dropna(subset=["hotel_id", "booking_created_date"])
    df["is_STR"] = parse_bool_series(df["is_STR"])
    for metric_name in METRICS:
        df[metric_name] = parse_number_series(df[metric_name])
    df["month"] = df["booking_created_date"].dt.to_period("M").dt.to_timestamp()
    df["month_num"] = df["booking_created_date"].dt.month
    hotel_attrs = (
        df.groupby("hotel_id", as_index=False)
        .agg({"is_STR": mode_value})
    )
    hotel_attrs["segment_key"] = hotel_attrs["is_STR"].apply(make_segment_key)
    monthly_values = (
        df.groupby(["hotel_id", "month", "month_num"], as_index=False)
        [list(METRICS.keys())]
        .sum()
    )
    monthly_counts = (
        df.groupby(["hotel_id", "month"], as_index=False)
        .size()
        .rename(columns={"size": "source_rows"})
    )
    monthly = monthly_values.merge(
        monthly_counts, on=["hotel_id", "month"], how="left"
    )
    min_month = df["month"].min().to_period("M")
    max_month = df["month"].max().to_period("M")
    all_months = pd.period_range(min_month, max_month, freq="M").to_timestamp()
    hotel_ids = hotel_attrs["hotel_id"].unique()
    grid = pd.MultiIndex.from_product(
        [hotel_ids, all_months], names=["hotel_id", "month"]
    ).to_frame(index=False)
    grid["month_num"] = grid["month"].dt.month
    full_monthly = grid.merge(hotel_attrs, on="hotel_id", how="left")
    full_monthly = full_monthly.merge(
        monthly, on=["hotel_id", "month", "month_num"], how="left"
    )
    full_monthly["source_rows"] = full_monthly["source_rows"].fillna(0).astype(int)
    for metric_name in METRICS:
        full_monthly[metric_name] = full_monthly[metric_name].fillna(0)
    full_monthly["has_source_row"] = full_monthly["source_rows"] > 0
    full_monthly["has_any_positive_metric"] = full_monthly[list(METRICS.keys())].gt(0).any(axis=1)
    full_monthly["has_observation"] = full_monthly["has_source_row"] | full_monthly["has_any_positive_metric"]
    observed_bounds = (
        full_monthly[full_monthly["has_observation"]]
        .groupby("hotel_id", as_index=False)
        .agg(
            first_observed_month=("month", "min"),
            last_observed_month=("month", "max"),
            observed_months=("month", "nunique")
        )
    )
    full_monthly = full_monthly.merge(observed_bounds, on="hotel_id", how="left")
    full_monthly["in_observed_span"] = (
        full_monthly["first_observed_month"].notna()
        & (full_monthly["month"] >= full_monthly["first_observed_month"])
        & (full_monthly["month"] <= full_monthly["last_observed_month"])
    )
    hotel_attrs = hotel_attrs.merge(observed_bounds, on="hotel_id", how="left")
    return df, full_monthly, hotel_attrs


@st.cache_data
def prepare_actions_data(actions_df):
    actions_df = normalize_columns(actions_df)
    check_columns(actions_df, ACTION_COLUMNS, "Action Logs")
    df = actions_df.copy()
    df["hotel_id"] = df["hotel_id"].astype(str).str.strip()
    df["action_date"] = pd.to_datetime(df["action_date"], errors="coerce")
    df["subject"] = df["subject"].astype(str).str.strip()
    df["outcome"] = df["outcome"].astype(str).str.strip()
    df = df.dropna(subset=["hotel_id", "action_date"]).reset_index(drop=True)

    df["action_id"] = np.arange(1, len(df) + 1)
    df["action_month"] = df["action_date"].dt.to_period("M").dt.to_timestamp()
    df["outcome_group"] = df["outcome"].apply(classify_outcome_group)
    return df
