import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import SCIPY_AVAILABLE, scipy_stats, SUCCESS_OUTCOMES, NEGATIVE_OUTCOMES
from action_log_effect_config import SUBJECT_EFFECT_MAP
from action_log_effect_calculator import (
    assign_effect_type,
    get_analysis_months,
    get_effective_month,
    normalize_subject,
)


CANONICAL_SUBJECT_BY_NORMALIZED = {
    normalize_subject(subject): subject
    for subject in SUBJECT_EFFECT_MAP
}


def _normal_two_sided_pvalue_from_t(t_value):
    if pd.isna(t_value):
        return np.nan
    z = abs(float(t_value))
    return 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))


def _betacf(a, b, x, max_iter=200, eps=3e-14):
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _regularized_beta(x, a, b):
    if pd.isna(x) or x < 0 or x > 1:
        return np.nan
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _student_t_two_sided_pvalue(t_value, df):
    if pd.isna(t_value) or pd.isna(df) or df <= 0:
        return np.nan
    if np.isinf(t_value):
        return 0.0
    t_abs = abs(float(t_value))
    x = float(df) / (float(df) + t_abs * t_abs)
    p_value = _regularized_beta(x, float(df) / 2.0, 0.5)
    if pd.isna(p_value):
        return np.nan
    return float(min(max(p_value, 0.0), 1.0))


def one_sample_ttest(values, mu=0.0):
    arr = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().astype(float).values
    n = len(arr)
    if n < 2:
        return np.nan, np.nan, np.nan, n, np.nan, np.nan, "Недостаточно данных"
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    if std == 0:
        t_stat = 0.0 if abs(mean - mu) < 1e-12 else np.inf
        df = n - 1
        p_value = 1.0 if t_stat == 0 else 0.0
    else:
        se = std / math.sqrt(n)
        t_stat = (mean - mu) / se
        df = n - 1
        if SCIPY_AVAILABLE:
            p_value = float(2 * scipy_stats.t.sf(abs(t_stat), df))
        else:
            p_value = float(_student_t_two_sided_pvalue(t_stat, df))
    return mean, std, t_stat, n, df, p_value, "scipy" if SCIPY_AVAILABLE else "normal approx"


def welch_ttest(values_a, values_b):
    a = pd.Series(values_a).replace([np.inf, -np.inf], np.nan).dropna().astype(float).values
    b = pd.Series(values_b).replace([np.inf, -np.inf], np.nan).dropna().astype(float).values
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return np.nan, np.nan, np.nan, np.nan, n_a, n_b, np.nan, np.nan, np.nan, "Недостаточно данных"
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a, var_b = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    se2 = var_a / n_a + var_b / n_b
    if se2 <= 0:
        t_stat = 0.0 if abs(mean_a - mean_b) < 1e-12 else np.inf
        df = n_a + n_b - 2
        p_value = 1.0 if t_stat == 0 else 0.0
    else:
        t_stat = (mean_a - mean_b) / math.sqrt(se2)
        numerator = se2 ** 2
        denominator = ((var_a / n_a) ** 2 / (n_a - 1)) + ((var_b / n_b) ** 2 / (n_b - 1))
        df = numerator / denominator if denominator > 0 else n_a + n_b - 2
        if SCIPY_AVAILABLE:
            p_value = float(2 * scipy_stats.t.sf(abs(t_stat), df))
        else:
            p_value = float(_student_t_two_sided_pvalue(t_stat, df))
    return mean_a, mean_b, mean_a - mean_b, t_stat, n_a, n_b, df, p_value, math.sqrt(se2) if se2 > 0 else np.nan, "scipy" if SCIPY_AVAILABLE else "normal approx"


def format_pvalue(p):
    if pd.isna(p):
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def normalize_month(value):
    if value is None or pd.isna(value):
        return pd.NaT
    if isinstance(value, pd.Period):
        return value.asfreq("M")
    text = str(value).strip()
    if not text:
        return pd.NaT
    if "." in text and "-" not in text:
        text = text.replace(".", "-")
    return pd.to_datetime(text, errors="coerce").to_period("M")


def paired_ttest(before_values, after_values):
    before = pd.Series(before_values).replace([np.inf, -np.inf], np.nan).astype(float)
    after = pd.Series(after_values).replace([np.inf, -np.inf], np.nan).astype(float)
    valid = before.notna() & after.notna()
    before = before[valid]
    after = after[valid]
    n = int(len(before))
    if n < 2:
        return np.nan, np.nan, n, np.nan
    if SCIPY_AVAILABLE:
        test = scipy_stats.ttest_rel(before, after, nan_policy="omit")
        t_stat = float(test.statistic)
        p_value = float(test.pvalue)
    else:
        diff = before - after
        std = float(diff.std(ddof=1))
        if std == 0:
            mean_diff = float(diff.mean())
            t_stat = 0.0 if abs(mean_diff) < 1e-12 else math.copysign(np.inf, mean_diff)
            p_value = 1.0 if t_stat == 0 else 0.0
        else:
            t_stat = float(diff.mean() / (std / math.sqrt(n)))
            p_value = float(_student_t_two_sided_pvalue(t_stat, n - 1))
    df = n - 1
    if not 0 <= p_value <= 1:
        p_value = np.nan
    return t_stat, p_value, n, df


def _platform_status(n, p_value, delta_abs, alpha, min_observations):
    if n < min_observations or pd.isna(p_value):
        return "insufficient_data", "Недостаточно наблюдений для устойчивого t-test."
    if p_value < alpha and delta_abs > 0:
        return "significant_positive", "Статистически значимый рост GBB после действия."
    if p_value < alpha and delta_abs < 0:
        return (
            "significant_negative",
            "Статистически значимое снижение GBB после действия. Требуется дополнительная проверка перед интерпретацией.",
        )
    if delta_abs > 0:
        return "not_significant_positive_trend", "Средний GBB вырос, но статистической значимости недостаточно."
    if delta_abs < 0:
        return "not_significant_negative_trend", "Средний GBB снизился, но статистической значимости недостаточно."
    return "not_significant_neutral", "Средний GBB не изменился."


def _direction(delta_abs):
    if pd.isna(delta_abs):
        return "unknown"
    if delta_abs > 0:
        return "positive"
    if delta_abs < 0:
        return "negative"
    return "neutral"


def _prepare_gbb_monthly(bookings_df):
    if bookings_df is None or bookings_df.empty:
        return pd.DataFrame(columns=["hotel_id", "month", "gbb"])
    df = bookings_df.copy()
    if "month" not in df.columns:
        if "calendar_month" in df.columns:
            df["month"] = df["calendar_month"]
        elif "booking_created_date" in df.columns:
            df["month"] = df["booking_created_date"]
        else:
            raise ValueError("Для t-test нужна колонка month, calendar_month или booking_created_date")
    if "has_source_row" in df.columns:
        df = df[df["has_source_row"].fillna(False).astype(bool)].copy()
    elif "has_metric_observation" in df.columns:
        df = df[df["has_metric_observation"].fillna(False).astype(bool)].copy()
    df["hotel_id"] = df["hotel_id"].astype("string").str.strip()
    df["month"] = df["month"].map(normalize_month).astype("period[M]")
    df["gbb"] = pd.to_numeric(df["gbb"], errors="coerce")
    df = df.dropna(subset=["hotel_id", "month"])
    return (
        df.groupby(["hotel_id", "month"], as_index=False)
        .agg(gbb=("gbb", lambda values: values.sum(min_count=1)))
    )


def build_manager_action_ttest(
    bookings_df,
    actions_df,
    allowed_outcomes=("Published", "Fixed", "Returned"),
    alpha=0.05,
    min_observations=20,
    dedup_enabled=True,
):
    calculated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    monthly = _prepare_gbb_monthly(bookings_df)
    actions = actions_df.copy() if actions_df is not None else pd.DataFrame()
    if actions.empty:
        empty_results = pd.DataFrame()
        empty_pairs = pd.DataFrame()
        run_summary = {
            "source": "streamlit",
            "hotels_count": int(monthly["hotel_id"].nunique()) if not monthly.empty else 0,
            "actions_count": 0,
            "valid_pairs_before_dedup": 0,
            "valid_pairs_count": 0,
            "duplicate_pairs_removed": 0,
            "excluded_pairs_count": 0,
            "subjects_total": 0,
            "subjects_tested": 0,
            "subjects_significant": 0,
            "dedup_enabled": dedup_enabled,
            "min_observations": min_observations,
            "alpha": alpha,
            "created_at": calculated_at,
        }
        return empty_results, empty_pairs, pd.DataFrame(), run_summary

    for column in ["hotel_id", "action_date", "subject", "outcome"]:
        if column not in actions.columns:
            actions[column] = pd.NA
    actions["hotel_id"] = actions["hotel_id"].astype("string").str.strip()
    actions["subject"] = actions["subject"].astype("string").str.strip()
    actions["subject_normalized"] = actions["subject"].map(normalize_subject)
    actions["subject_canonical"] = actions["subject_normalized"].map(CANONICAL_SUBJECT_BY_NORMALIZED).fillna(actions["subject"])
    actions["outcome"] = actions["outcome"].astype("string").str.strip()
    if allowed_outcomes is not None:
        actions = actions[actions["outcome"].isin(list(allowed_outcomes))].copy()
    actions["action_date"] = pd.to_datetime(actions["action_date"], errors="coerce")
    actions["effect_window"] = actions["subject"].apply(assign_effect_type)
    actions["effective_month"] = pd.Series(pd.NaT, index=actions.index).dt.to_period("M")
    valid_dates = actions["action_date"].notna()
    actions.loc[valid_dates, "effective_month"] = actions.loc[valid_dates, "action_date"].map(get_effective_month)
    actions["month_before"] = pd.Series(pd.NaT, index=actions.index).dt.to_period("M")
    actions["month_after"] = pd.Series(pd.NaT, index=actions.index).dt.to_period("M")
    known_window = actions["effect_window"].ne("unknown") & actions["effective_month"].notna()
    for index, row in actions.loc[known_window, ["effective_month", "effect_window"]].iterrows():
        before_month, after_month = get_analysis_months(row["effective_month"], row["effect_window"])
        actions.at[index, "month_before"] = before_month
        actions.at[index, "month_after"] = after_month

    before = monthly.rename(columns={"month": "month_before", "gbb": "gbb_before"})
    after = monthly.rename(columns={"month": "month_after", "gbb": "gbb_after"})
    pairs = actions.merge(before, on=["hotel_id", "month_before"], how="left")
    pairs = pairs.merge(after, on=["hotel_id", "month_after"], how="left")
    pairs["pair_status"] = "valid"
    pairs["exclusion_reason"] = None
    pairs.loc[pairs["hotel_id"].isna() | pairs["hotel_id"].eq(""), ["pair_status", "exclusion_reason"]] = ["excluded", "missing_hotel_id"]
    pairs.loc[pairs["subject"].isna() | pairs["subject"].eq(""), ["pair_status", "exclusion_reason"]] = ["excluded", "missing_subject"]
    pairs.loc[pairs["effect_window"].eq("unknown") & pairs["exclusion_reason"].isna(), ["pair_status", "exclusion_reason"]] = ["excluded", "missing_effect_window_config"]
    pairs.loc[pairs["action_date"].isna() & pairs["exclusion_reason"].isna(), ["pair_status", "exclusion_reason"]] = ["excluded", "invalid_action_date"]
    missing_before = pairs["gbb_before"].isna()
    missing_after = pairs["gbb_after"].isna()
    pairs.loc[missing_before & missing_after & pairs["exclusion_reason"].isna(), ["pair_status", "exclusion_reason"]] = ["excluded", "missing_both"]
    pairs.loc[missing_before & ~missing_after & pairs["exclusion_reason"].isna(), ["pair_status", "exclusion_reason"]] = ["excluded", "missing_gbb_before"]
    pairs.loc[~missing_before & missing_after & pairs["exclusion_reason"].isna(), ["pair_status", "exclusion_reason"]] = ["excluded", "missing_gbb_after"]
    pairs.loc[pairs["gbb_before"].replace([np.inf, -np.inf], np.nan).isna() & pairs["exclusion_reason"].isna(), ["pair_status", "exclusion_reason"]] = ["excluded", "invalid_gbb_value"]
    pairs.loc[pairs["gbb_after"].replace([np.inf, -np.inf], np.nan).isna() & pairs["exclusion_reason"].isna(), ["pair_status", "exclusion_reason"]] = ["excluded", "invalid_gbb_value"]

    valid_pairs_raw_count = int(pairs["pair_status"].eq("valid").sum())
    excluded_pairs_count = int(pairs["pair_status"].eq("excluded").sum())
    valid_pairs = pairs[pairs["pair_status"].eq("valid")].copy()
    valid_pairs["dedup_key"] = (
        valid_pairs["hotel_id"].astype(str) + "|" + valid_pairs["subject_normalized"].astype(str) + "|"
        + valid_pairs["effective_month"].astype(str) + "|" + valid_pairs["month_before"].astype(str)
        + "|" + valid_pairs["month_after"].astype(str)
    )
    if dedup_enabled:
        valid_pairs = valid_pairs.drop_duplicates("dedup_key", keep="first").copy()
    duplicate_pairs_removed = valid_pairs_raw_count - int(len(valid_pairs))

    rows = []
    for (subject, effect_window), group in valid_pairs.groupby(["subject_canonical", "effect_window"], dropna=False):
        n = int(len(group))
        mean_before = float(group["gbb_before"].mean()) if n else np.nan
        mean_after = float(group["gbb_after"].mean()) if n else np.nan
        delta_abs = mean_after - mean_before if n else np.nan
        delta_pct = delta_abs / mean_before * 100 if mean_before and not pd.isna(mean_before) else np.nan
        t_stat, p_value, _, df = paired_ttest(group["gbb_before"], group["gbb_after"])
        if n < min_observations:
            t_stat = np.nan
            p_value = np.nan
            df = np.nan
        status, summary = _platform_status(n, p_value, delta_abs, alpha, min_observations)
        rows.append({
            "subject": subject,
            "effect_window": effect_window,
            "n": n,
            "mean_before": mean_before,
            "mean_after": mean_after,
            "delta_abs": delta_abs,
            "delta_pct": delta_pct,
            "t_statistic": t_stat,
            "df": df,
            "p_value_two_tail": p_value,
            "alpha": alpha,
            "is_significant": bool(pd.notna(p_value) and p_value < alpha),
            "direction": _direction(delta_abs),
            "platform_status": status,
            "platform_summary": summary,
            "period_min": str(min(group["month_before"].min(), group["month_after"].min())),
            "period_max": str(max(group["month_before"].max(), group["month_after"].max())),
            "calculated_at": calculated_at,
        })

    results = pd.DataFrame(rows)
    if not results.empty:
        order = {
            "significant_positive": 0,
            "significant_negative": 1,
            "not_significant_positive_trend": 2,
            "not_significant_negative_trend": 3,
            "not_significant_neutral": 4,
            "insufficient_data": 5,
        }
        results["status_order"] = results["platform_status"].map(order).fillna(99)
        results = results.sort_values(["status_order", "p_value_two_tail", "subject"], na_position="last").drop(columns=["status_order"])

    diagnostics = (
        pairs.loc[pairs["pair_status"].eq("excluded"), "exclusion_reason"]
        .fillna("unknown")
        .value_counts()
        .rename_axis("exclusion_reason")
        .reset_index(name="actions_count")
    )
    run_summary = {
        "source": "streamlit",
        "hotels_count": int(monthly["hotel_id"].nunique()) if not monthly.empty else 0,
        "actions_count": int(len(actions)),
        "valid_pairs_before_dedup": valid_pairs_raw_count,
        "valid_pairs_count": int(len(valid_pairs)),
        "duplicate_pairs_removed": duplicate_pairs_removed,
        "excluded_pairs_count": excluded_pairs_count,
        "subjects_total": int(actions["subject_normalized"].nunique()),
        "subjects_tested": int((results["n"] >= min_observations).sum()) if not results.empty else 0,
        "subjects_significant": int(results["is_significant"].sum()) if not results.empty else 0,
        "dedup_enabled": dedup_enabled,
        "min_observations": min_observations,
        "alpha": alpha,
        "created_at": calculated_at,
    }
    return results, pairs, diagnostics, run_summary


def pvalue_interpretation(p_value, t_stat, min_n_ok=True):
    if not min_n_ok or pd.isna(p_value):
        return "Недостаточно данных для статистического вывода"
    if p_value < 0.05:
        direction = "положительный" if t_stat > 0 else "отрицательный"
        return f"Статистически заметный {direction} эффект на уровне 5%"
    return "Нет статистически значимого отличия от нуля/между группами на уровне 5%"


def build_ttest_tables(period_classification, action_impact):
    rows = []
    if not action_impact.empty and "manager_effect_pp" in action_impact.columns:
        clean_actions = action_impact.dropna(subset=["manager_effect_pp"]).copy()
        mean, std, t_stat, n, df, p, method = one_sample_ttest(clean_actions["manager_effect_pp"], 0)
        rows.append({
            "Проверка": "Manager effect после Action Logs отличается от 0",
            "Тип теста": "One-sample t-test",
            "Группа A": "Все рассчитанные Action Logs",
            "Группа B / baseline": "0 п.п.",
            "n A": n, "n B": np.nan,
            "Среднее A": mean, "Среднее B": 0,
            "Разница средних": mean,
            "t-statistic": t_stat, "df": df, "p-value": p,
            "Метод p-value": method,
            "Интерпретация": pvalue_interpretation(p, t_stat, n >= 2)
        })
        if "outcome" in clean_actions.columns:
            success = clean_actions[clean_actions["outcome"].isin(SUCCESS_OUTCOMES)]["manager_effect_pp"]
            refused = clean_actions[clean_actions["outcome"].isin(NEGATIVE_OUTCOMES)]["manager_effect_pp"]
            mean_a, mean_b, diff, t_stat, n_a, n_b, df, p, se, method = welch_ttest(success, refused)
            rows.append({
                "Проверка": "Успешные outcomes дают больший manager effect, чем негативные",
                "Тип теста": "Welch two-sample t-test",
                "Группа A": "Published/Fixed/Returned/Resolved",
                "Группа B / baseline": "Refused/Impossible/Negative",
                "n A": n_a, "n B": n_b,
                "Среднее A": mean_a, "Среднее B": mean_b,
                "Разница средних": diff,
                "t-statistic": t_stat, "df": df, "p-value": p,
                "Метод p-value": method,
                "Интерпретация": pvalue_interpretation(p, t_stat, n_a >= 2 and n_b >= 2)
            })
        if "has_overlap" in clean_actions.columns:
            no_overlap = clean_actions[clean_actions["has_overlap"] == False]["manager_effect_pp"]
            overlap = clean_actions[clean_actions["has_overlap"] == True]["manager_effect_pp"]
            mean_a, mean_b, diff, t_stat, n_a, n_b, df, p, se, method = welch_ttest(no_overlap, overlap)
            rows.append({
                "Проверка": "Действия без наложений отличаются от действий с наложениями",
                "Тип теста": "Welch two-sample t-test",
                "Группа A": "Без наложений",
                "Группа B / baseline": "С наложениями",
                "n A": n_a, "n B": n_b,
                "Среднее A": mean_a, "Среднее B": mean_b,
                "Разница средних": diff,
                "t-statistic": t_stat, "df": df, "p-value": p,
                "Метод p-value": method,
                "Интерпретация": pvalue_interpretation(p, t_stat, n_a >= 2 and n_b >= 2)
            })
    if not period_classification.empty and "individual_effect_pp" in period_classification.columns:
        pc = period_classification.dropna(subset=["individual_effect_pp"]).copy()
        with_actions = pc[pc["actions_count"] > 0]["individual_effect_pp"] if "actions_count" in pc.columns else pd.Series(dtype=float)
        without_actions = pc[pc["actions_count"] == 0]["individual_effect_pp"] if "actions_count" in pc.columns else pd.Series(dtype=float)
        mean_a, mean_b, diff, t_stat, n_a, n_b, df, p, se, method = welch_ttest(with_actions, without_actions)
        rows.append({
            "Проверка": "Объект-периоды с Action Logs отличаются от объект-периодов без Action Logs",
            "Тип теста": "Welch two-sample t-test",
            "Группа A": "Есть Action Logs",
            "Группа B / baseline": "Нет Action Logs",
            "n A": n_a, "n B": n_b,
            "Среднее A": mean_a, "Среднее B": mean_b,
            "Разница средних": diff,
            "t-statistic": t_stat, "df": df, "p-value": p,
            "Метод p-value": method,
            "Интерпретация": pvalue_interpretation(p, t_stat, n_a >= 2 and n_b >= 2)
        })
        mean, std, t_stat, n, df, p, method = one_sample_ttest(pc["individual_effect_pp"], 0)
        rows.append({
            "Проверка": "Средний individual effect по выборке отличается от 0",
            "Тип теста": "One-sample t-test",
            "Группа A": "Все объект-периоды",
            "Группа B / baseline": "0 п.п.",
            "n A": n, "n B": np.nan,
            "Среднее A": mean, "Среднее B": 0,
            "Разница средних": mean,
            "t-statistic": t_stat, "df": df, "p-value": p,
            "Метод p-value": method,
            "Интерпретация": pvalue_interpretation(p, t_stat, n >= 2)
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        for c in ["Среднее A", "Среднее B", "Разница средних", "t-statistic", "df", "p-value"]:
            if c in result.columns:
                result[c] = pd.to_numeric(result[c], errors="coerce")
    return result
