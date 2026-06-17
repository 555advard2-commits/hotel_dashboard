import re

import numpy as np
import pandas as pd


def parse_bool(value):
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    true_values = ["true", "1", "yes", "y", "да", "истина", "str", "апарт", "апарт-отель"]
    false_values = ["false", "0", "no", "n", "нет", "ложь", "hotel", "отель"]
    if text in true_values:
        return True
    if text in false_values:
        return False
    return False


def parse_number(value):
    if pd.isna(value):
        return 0.0
    text = str(value).strip()
    text = text.replace("\xa0", "")
    text = text.replace(" ", "")
    text = re.sub(r"[^0-9,\.\-]", "", text)
    if text == "":
        return 0.0
    if "," in text and "." in text:
        text = text.replace(",", "")
    elif "," in text and "." not in text:
        text = text.replace(",", ".")
    try:
        return float(text)
    except Exception:
        return 0.0


def parse_number_series(series):
    text = series.astype("string").fillna("")
    text = text.str.strip()
    text = text.str.replace("\xa0", "", regex=False)
    text = text.str.replace(" ", "", regex=False)
    text = text.str.replace(r"[^0-9,\.\-]", "", regex=True)
    both_separators = text.str.contains(",", regex=False) & text.str.contains(".", regex=False)
    comma_decimal = text.str.contains(",", regex=False) & ~text.str.contains(".", regex=False)
    text = text.mask(both_separators, text.str.replace(",", "", regex=False))
    text = text.mask(comma_decimal, text.str.replace(",", ".", regex=False))
    return pd.to_numeric(text, errors="coerce").fillna(0.0)


def parse_bool_series(series):
    text = series.astype("string").fillna("").str.strip().str.lower()
    true_values = {
        "true", "1", "yes", "y", "str",
        "\u0434\u0430", "\u0438\u0441\u0442\u0438\u043d\u0430",
        "\u0430\u043f\u0430\u0440\u0442", "\u0430\u043f\u0430\u0440\u0442-\u043e\u0442\u0435\u043b\u044c",
    }
    return text.isin(true_values)


def safe_divide(a, b):
    if b == 0 or pd.isna(b):
        return np.nan
    return a / b


def mode_value(series):
    clean = series.dropna()
    if clean.empty:
        return "UNKNOWN"
    mode = clean.mode()
    if len(mode) > 0:
        return mode.iloc[0]
    return clean.iloc[0]


def format_number(value, digits=0):
    if pd.isna(value):
        return "—"
    text = f"{value:,.{digits}f}"
    return text.replace(",", " ")


def format_percent(value, digits=2):
    if pd.isna(value):
        return "—"
    return f"{value:.{digits}f}%"


def classify_outcome_group(outcome):
    from config import SUCCESS_OUTCOMES, NEGATIVE_OUTCOMES, IN_PROGRESS_OUTCOMES
    text = str(outcome).strip()
    if text in SUCCESS_OUTCOMES:
        return "Успешный / изменяющий"
    if text in NEGATIVE_OUTCOMES:
        return "Отказ / отрицательный"
    if text in IN_PROGRESS_OUTCOMES:
        return "В процессе / follow-up"
    return "Другое"


def make_segment_key(is_str):
    if bool(is_str):
        return "STR / апарт-отели"
    return "HOTEL / обычные отели"


def add_months(timestamp, number_of_months):
    period = timestamp.to_period("M")
    new_period = period + number_of_months
    return new_period.to_timestamp()


def make_period_column(date_series, grain):
    if grain == "Месяц":
        return date_series.dt.to_period("M").dt.to_timestamp()
    if grain == "Квартал":
        return date_series.dt.to_period("Q").dt.to_timestamp()
    return date_series.dt.to_period("Y").dt.to_timestamp()


def make_period_label(date_series, grain):
    if grain == "Месяц":
        return date_series.dt.to_period("M").astype(str)
    if grain == "Квартал":
        return date_series.dt.to_period("Q").astype(str)
    return date_series.dt.to_period("Y").astype(str)


def month_diff_inclusive(start_month, end_month):
    if pd.isna(start_month) or pd.isna(end_month):
        return 0
    start_period = pd.Timestamp(start_month).to_period("M")
    end_period = pd.Timestamp(end_month).to_period("M")
    return int(end_period.ordinal - start_period.ordinal + 1)


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def normalize_columns(df):
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df
