import numpy as np
import pandas as pd
import plotly.express as px

from config import METRICS, SUCCESS_OUTCOMES
from action_log_effect_calculator import assign_effect_type, get_analysis_months, get_effective_month
from action_log_effect_config import EFFECT_TYPE_LABELS
from utils import (
    safe_divide, month_diff_inclusive, make_period_column,
    make_period_label, add_months, mode_value, classify_outcome_group
)


def get_analysis_window(monthly_df, mode, custom_start=None, custom_end=None):
    available_months = sorted(monthly_df["month"].dropna().unique())
    if len(available_months) == 0:
        return None, None
    min_month = pd.Timestamp(available_months[0])
    max_month = pd.Timestamp(available_months[-1])
    if mode == "Последние 24 месяца":
        start_month = (max_month.to_period("M") - 23).to_timestamp()
        end_month = max_month
    elif mode == "Последние 12 месяцев":
        start_month = (max_month.to_period("M") - 11).to_timestamp()
        end_month = max_month
    elif mode == "Пользовательский диапазон" and custom_start is not None and custom_end is not None:
        start_month = pd.Timestamp(custom_start).to_period("M").to_timestamp()
        end_month = pd.Timestamp(custom_end).to_period("M").to_timestamp()
    else:
        start_month = min_month
        end_month = max_month
    if start_month < min_month:
        start_month = min_month
    if end_month > max_month:
        end_month = max_month
    if start_month > end_month:
        start_month = min_month
        end_month = max_month
    return start_month, end_month


def _build_hotel_quality_row(
    group, window_start, window_end, total_window_months,
    selected_metrics, min_nonzero_months, max_zero_months,
    max_zero_share_pct, min_observed_months,
    require_full_observed_span, strict_all_selected_metrics
):
    hotel_id = group["hotel_id"].iloc[0]
    segment_key = group["segment_key"].iloc[0]
    is_str = bool(group["is_STR"].iloc[0])
    first_observed = group["first_observed_month"].dropna()
    last_observed = group["last_observed_month"].dropna()
    first_observed_month = first_observed.iloc[0] if not first_observed.empty else pd.NaT
    last_observed_month = last_observed.iloc[0] if not last_observed.empty else pd.NaT
    in_span = group["in_observed_span"].fillna(False)
    observed_span_months_in_window = int(in_span.sum())
    source_row_months = int(
        group["has_source_row"].fillna(False).sum()
        if "has_source_row" in group.columns
        else group["has_observation"].fillna(False).sum()
    )
    selected_positive_by_month = group[selected_metrics].gt(0)
    if strict_all_selected_metrics:
        good_month_mask = selected_positive_by_month.all(axis=1) & in_span
    else:
        good_month_mask = selected_positive_by_month.any(axis=1) & in_span
    nonzero_months_selected = int(good_month_mask.sum())
    zero_months_inside_span = int(((~good_month_mask) & in_span).sum())
    zero_share_inside_span = safe_divide(zero_months_inside_span, observed_span_months_in_window)
    zero_share_inside_span_pct = zero_share_inside_span * 100 if not pd.isna(zero_share_inside_span) else np.nan
    metric_nonzero = {}
    metric_zero_inside = {}
    for metric_name in selected_metrics:
        metric_nonzero[metric_name] = int(((group[metric_name] > 0) & in_span).sum())
        metric_zero_inside[metric_name] = int(((group[metric_name] <= 0) & in_span).sum())
    covers_window = (
        pd.notna(first_observed_month)
        and pd.notna(last_observed_month)
        and first_observed_month <= window_start
        and last_observed_month >= window_end
    )
    full_window_months_with_data = int(
        group["has_source_row"].fillna(False).sum()
        if "has_source_row" in group.columns
        else group["has_observation"].fillna(False).sum()
    )
    has_data_for_full_window = full_window_months_with_data == total_window_months
    reasons = []
    if require_full_observed_span and not has_data_for_full_window:
        reasons.append("отель не покрывает всё выбранное окно наблюдения")
    if observed_span_months_in_window < min_observed_months:
        reasons.append(f"мало месяцев в активном периоде: {observed_span_months_in_window}")
    if nonzero_months_selected < min_nonzero_months:
        reasons.append(f"мало месяцев без нулей по выбранным метрикам: {nonzero_months_selected}")
    if zero_months_inside_span > max_zero_months:
        reasons.append(f"слишком много нулевых месяцев: {zero_months_inside_span}")
    if not pd.isna(zero_share_inside_span_pct) and zero_share_inside_span_pct > max_zero_share_pct:
        reasons.append(f"слишком высокая доля нулей: {zero_share_inside_span_pct:.1f}%")
    is_in_sample = len(reasons) == 0
    row = {
        "hotel_id": hotel_id,
        "is_STR": is_str,
        "segment_key": segment_key,
        "first_observed_month": first_observed_month,
        "last_observed_month": last_observed_month,
        "window_start": window_start,
        "window_end": window_end,
        "window_months": total_window_months,
        "observed_span_months_in_window": observed_span_months_in_window,
        "source_row_months": source_row_months,
        "full_window_months_with_data": full_window_months_with_data,
        "nonzero_months_selected": nonzero_months_selected,
        "zero_months_inside_span": zero_months_inside_span,
        "zero_share_inside_span_pct": zero_share_inside_span_pct,
        "covers_full_window": bool(has_data_for_full_window),
        "is_in_sample": bool(is_in_sample),
        "exclusion_reason": "; ".join(reasons)
    }
    for metric_name in selected_metrics:
        row[f"nonzero_months_{metric_name}"] = metric_nonzero[metric_name]
        row[f"zero_months_{metric_name}"] = metric_zero_inside[metric_name]
    return row


def evaluate_sample_quality(
    monthly_df, selected_metrics, window_start, window_end,
    require_full_observed_span, min_nonzero_months, max_zero_months,
    max_zero_share_pct, min_observed_months, strict_all_selected_metrics
):
    if monthly_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    selected_metrics = [m for m in selected_metrics if m in monthly_df.columns]
    if not selected_metrics:
        selected_metrics = ["gbb"]
    window = monthly_df[
        (monthly_df["month"] >= window_start)
        & (monthly_df["month"] <= window_end)
    ].copy()
    total_window_months = month_diff_inclusive(window_start, window_end)
    if window.empty or total_window_months <= 0:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    for _, group in window.groupby("hotel_id", sort=False):
        group = group.sort_values("month")
        row = _build_hotel_quality_row(
            group, window_start, window_end, total_window_months,
            selected_metrics, min_nonzero_months, max_zero_months,
            max_zero_share_pct, min_observed_months,
            require_full_observed_span, strict_all_selected_metrics
        )
        rows.append(row)
    quality = pd.DataFrame(rows)
    if quality.empty:
        return quality, window.iloc[0:0].copy()
    kept_hotels = set(quality[quality["is_in_sample"]]["hotel_id"].astype(str))
    prepared_monthly = window[
        window["hotel_id"].astype(str).isin(kept_hotels)
        & window["in_observed_span"].fillna(False)
    ].copy()
    return quality, prepared_monthly


def make_sample_quality_summary(quality_table):
    if quality_table.empty:
        return pd.DataFrame()
    total_hotels = quality_table["hotel_id"].nunique()
    kept_hotels = int(quality_table["is_in_sample"].sum())
    excluded_hotels = total_hotels - kept_hotels
    reason_rows = []
    excluded = quality_table[~quality_table["is_in_sample"]].copy()
    for reason in [
        "отель не покрывает всё выбранное окно наблюдения",
        "мало месяцев в активном периоде",
        "мало месяцев без нулей",
        "слишком много нулевых месяцев",
        "слишком высокая доля нулей"
    ]:
        count = int(excluded["exclusion_reason"].str.contains(reason, na=False, regex=False).sum())
        reason_rows.append({
            "Причина отсечения": reason,
            "Отелей": count,
            "Доля от исходной базы, %": safe_divide(count, total_hotels) * 100
        })
    reason_rows.insert(0, {
        "Причина отсечения": "Оставлено в аналитической выборке",
        "Отелей": kept_hotels,
        "Доля от исходной базы, %": safe_divide(kept_hotels, total_hotels) * 100
    })
    reason_rows.insert(1, {
        "Причина отсечения": "Исключено всего",
        "Отелей": excluded_hotels,
        "Доля от исходной базы, %": safe_divide(excluded_hotels, total_hotels) * 100
    })
    return pd.DataFrame(reason_rows)


def create_sample_funnel_fig(summary_table):
    if summary_table.empty:
        return None
    rows = summary_table[summary_table["Причина отсечения"].isin([
        "Оставлено в аналитической выборке", "Исключено всего"
    ])].copy()
    fig = px.bar(
        rows, x="Причина отсечения", y="Отелей", text="Отелей",
        labels={"Причина отсечения": "", "Отелей": "Количество отелей"},
        title="Фильтр качества данных: сколько объектов осталось в выборке"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=420)
    return fig


def create_zero_distribution_fig(quality_table):
    if quality_table.empty:
        return None
    fig = px.histogram(
        quality_table, x="zero_share_inside_span_pct", nbins=30,
        labels={"zero_share_inside_span_pct": "Доля нулевых месяцев внутри активного периода, %", "count": "Количество отелей"},
        title="Распределение доли нулевых месяцев по отелям"
    )
    fig.update_layout(height=430)
    return fig


def create_observed_months_fig(quality_table):
    if quality_table.empty:
        return None
    fig = px.histogram(
        quality_table, x="observed_span_months_in_window", nbins=24,
        labels={"observed_span_months_in_window": "Месяцев активного периода внутри окна", "count": "Количество отелей"},
        title="Сколько месяцев наблюдения доступно по отелям"
    )
    fig.update_layout(height=430)
    return fig


def classify_period(row, threshold):
    expected = row.get("expected_value", 0)
    actual = row.get("actual_value", 0)
    individual = row.get("individual_effect_pct", np.nan)
    actions_count = row.get("actions_count", 0)
    if pd.isna(expected) or expected <= 0:
        return "Недостаточно данных"
    if pd.isna(individual):
        return "Недостаточно данных"
    if individual >= threshold and actions_count > 0:
        return "Рост выше локального фона, вероятен вклад менеджера"
    if individual >= threshold and actions_count == 0:
        return "Рост выше локального фона, но без Action Log"
    if abs(individual) < threshold and actual >= expected:
        return "Рост в основном сезонный / общий локальный фон"
    if abs(individual) < threshold and actual < expected:
        return "Динамика близка к фону, но ниже ожидания"
    if individual <= -threshold:
        return "Хуже локального фона / индивидуальная просадка"
    return "Смешанный эффект"


def calculate_period_classification(enriched_df, actions_df, metric_name, grain, threshold):
    df = enriched_df.copy()
    df["period"] = make_period_column(df["month"], grain)
    df["period_label"] = make_period_label(df["month"], grain)
    hotel_period = (
        df.groupby(["hotel_id", "period", "period_label", "is_STR", "segment_key"], as_index=False)
        .agg(
            actual_value=(metric_name, "sum"),
            expected_value=("expected_value", "sum")
        )
    )
    hotel_period["hotel_efficiency"] = hotel_period.apply(
        lambda row: safe_divide(row["actual_value"], row["expected_value"]), axis=1
    )
    hotel_period["hotel_residual_pct"] = hotel_period["hotel_efficiency"] - 1
    segment_period = (
        df.groupby(["segment_key", "period"], as_index=False)
        .agg(
            segment_actual=(metric_name, "sum"),
            segment_expected=("expected_value", "sum")
        )
    )
    segment_period["segment_efficiency"] = segment_period.apply(
        lambda row: safe_divide(row["segment_actual"], row["segment_expected"]), axis=1
    )
    segment_period["segment_residual_pct"] = segment_period["segment_efficiency"] - 1
    segment_period = segment_period.rename(
        columns={"segment_efficiency": "segment_efficiency_renamed"}
    )
    merged = hotel_period.merge(
        segment_period[["segment_key", "period", "segment_efficiency_renamed", "segment_residual_pct"]],
        on=["segment_key", "period"], how="left"
    )
    merged["segment_efficiency"] = merged["segment_efficiency_renamed"]
    merged["individual_effect_pct"] = merged["hotel_residual_pct"] - merged["segment_residual_pct"]
    if not actions_df.empty:
        act = actions_df.copy()
        act["period"] = make_period_column(act["action_month"], grain)
        counts = (
            act.groupby(["hotel_id", "period"], as_index=False)
            .size().rename(columns={"size": "actions_count"})
        )
        merged = merged.merge(counts, on=["hotel_id", "period"], how="left")
        merged["actions_count"] = merged["actions_count"].fillna(0).astype(int)
    else:
        merged["actions_count"] = 0
    merged["verdict"] = merged.apply(
        lambda row: classify_period(row, threshold), axis=1
    )
    cols = {
        "individual_effect_pct": "individual_effect_pp",
        "hotel_residual_pct": "hotel_residual_pp",
        "segment_residual_pct": "segment_residual_pp",
    }
    merged = merged.rename(columns=cols)
    return merged


def classify_action(row):
    effect = row["manager_effect_pp"]
    outcome = row["outcome"]
    if pd.isna(effect):
        return "Недостаточно данных"
    if outcome == "Refused" and effect > 5:
        return "Рост есть, но outcome Refused – не приписывать как принятый action"
    if effect >= 10:
        return "Положительный индивидуальный эффект после действия"
    if effect <= -10:
        return "После действия стало хуже относительно локального фона"
    if abs(effect) < 5:
        return "Явного индивидуального эффекта не видно"
    if effect > 0:
        return "Слабый положительный эффект"
    return "Слабый отрицательный эффект"



def _to_month_timestamp(value):
    if value is None or pd.isna(value):
        return pd.NaT
    if isinstance(value, pd.Period):
        return value.to_timestamp()
    return pd.Timestamp(value).to_period("M").to_timestamp()


def _get_action_effect_months(action_date, effect_type):
    parsed_date = pd.to_datetime(action_date, errors="coerce")
    if pd.isna(parsed_date) or effect_type == "unknown":
        return pd.NaT, pd.NaT, pd.NaT
    effective_period = get_effective_month(parsed_date)
    before_period, after_period = get_analysis_months(effective_period, effect_type)
    return (
        _to_month_timestamp(effective_period),
        _to_month_timestamp(before_period),
        _to_month_timestamp(after_period),
    )


def calculate_action_impact(
    actions_df, enriched_df, metric_name,
    pre_window_months, post_window_months, lag_months,
    min_months_per_side, overlap_policy
):
    if actions_df.empty:
        return pd.DataFrame()

    base = enriched_df.copy()
    actions_work = actions_df.copy().reset_index(drop=True)
    if "action_id" not in actions_work.columns:
        actions_work["action_id"] = np.arange(1, len(actions_work) + 1)

    actions_work["effect_type"] = actions_work["subject"].apply(assign_effect_type)
    actions_work["effect_type_label"] = actions_work["effect_type"].map(EFFECT_TYPE_LABELS).fillna(EFFECT_TYPE_LABELS["unknown"])
    effect_months = actions_work.apply(
        lambda row: _get_action_effect_months(row.get("action_date"), row.get("effect_type")),
        axis=1,
    )
    actions_work["effective_month"] = [months[0] for months in effect_months]
    actions_work["before_month"] = [months[1] for months in effect_months]
    actions_work["after_month"] = [months[2] for months in effect_months]

    hotel_lookup = base.set_index(["hotel_id", "month"])[
        [metric_name, "expected_value", "segment_key"]
    ]
    hotel_segment = (
        base.groupby("hotel_id", as_index=False)["segment_key"]
        .agg(mode_value)
        .rename(columns={"segment_key": "hotel_segment_key"})
    )
    group_month = (
        base.groupby(["segment_key", "month"], as_index=False)
        .agg(
            group_actual=(metric_name, "sum"),
            group_expected=("expected_value", "sum"),
        )
    )
    group_lookup = group_month.set_index(["segment_key", "month"])[
        ["group_actual", "group_expected"]
    ]

    rows = []
    for _, action in actions_work.iterrows():
        hotel_id = action["hotel_id"]
        action_id = action["action_id"]
        action_month = action["action_month"]
        effect_type = action["effect_type"]
        effect_type_label = action["effect_type_label"]
        effective_month = action["effective_month"]
        before_month = action["before_month"]
        after_month = action["after_month"]

        segment_row = hotel_segment[hotel_segment["hotel_id"] == hotel_id]
        segment_key = segment_row["hotel_segment_key"].iloc[0] if not segment_row.empty else ""
        pre_months = [before_month] if pd.notna(before_month) else []
        post_months = [after_month] if pd.notna(after_month) else []
        evaluation_months = pre_months + ([effective_month] if pd.notna(effective_month) else []) + post_months
        evaluation_start = min(evaluation_months) if evaluation_months else action_month
        evaluation_end = max(evaluation_months) if evaluation_months else action_month

        same_hotel_actions = actions_work[
            (actions_work["hotel_id"] == hotel_id)
            & (actions_work["action_id"] != action_id)
            & (actions_work["effective_month"] >= evaluation_start)
            & (actions_work["effective_month"] <= evaluation_end)
        ]
        overlapping_actions_count = int(len(same_hotel_actions))
        has_overlap = overlapping_actions_count > 0
        overlap_subjects = ", ".join(sorted(same_hotel_actions["subject"].dropna().astype(str).unique()[:5]))

        pre_hotel_index = pd.MultiIndex.from_product([[hotel_id], pre_months], names=["hotel_id", "month"])
        post_hotel_index = pd.MultiIndex.from_product([[hotel_id], post_months], names=["hotel_id", "month"])
        pre_group_index = pd.MultiIndex.from_product([[segment_key], pre_months], names=["segment_key", "month"])
        post_group_index = pd.MultiIndex.from_product([[segment_key], post_months], names=["segment_key", "month"])

        pre_hotel = hotel_lookup.reindex(pre_hotel_index).fillna(0)
        post_hotel = hotel_lookup.reindex(post_hotel_index).fillna(0)
        pre_group = group_lookup.reindex(pre_group_index).fillna(0)
        post_group = group_lookup.reindex(post_group_index).fillna(0)

        pre_months_available = int((pre_hotel["expected_value"] > 0).sum())
        post_months_available = int((post_hotel["expected_value"] > 0).sum())
        pre_actual = pre_hotel[metric_name].sum()
        pre_expected = pre_hotel["expected_value"].sum()
        post_actual = post_hotel[metric_name].sum()
        post_expected = post_hotel["expected_value"].sum()

        exclusion_reasons = []
        if segment_row.empty:
            exclusion_reasons.append("нет сегмента отеля для сравнения")
        if effect_type == "unknown":
            exclusion_reasons.append("тип эффекта не определен по subject")
        if pd.isna(effective_month):
            exclusion_reasons.append("некорректная дата Action Log")
        if not pre_months:
            exclusion_reasons.append("нет месяца ДО по методологии эффекта")
        if not post_months:
            exclusion_reasons.append("нет месяца ПОСЛЕ по методологии эффекта")
        if pre_months and pre_months_available < 1:
            exclusion_reasons.append(f"мало месяцев ДО: {pre_months_available} из 1")
        if post_months and post_months_available < 1:
            exclusion_reasons.append(f"мало месяцев ПОСЛЕ: {post_months_available} из 1")
        if pre_expected <= 0:
            exclusion_reasons.append("нулевое ожидание ДО")
        if post_expected <= 0:
            exclusion_reasons.append("нулевое ожидание ПОСЛЕ")
        if overlap_policy == "Исключать действия с наложениями" and has_overlap:
            exclusion_reasons.append("есть другие Action Logs в окне оценки")

        included_in_effect = len(exclusion_reasons) == 0
        pre_eff = safe_divide(pre_actual, pre_expected)
        post_eff = safe_divide(post_actual, post_expected)
        hotel_change = post_eff - pre_eff if not pd.isna(pre_eff) and not pd.isna(post_eff) else np.nan
        pre_peer_actual = pre_group["group_actual"].sum() - pre_actual
        pre_peer_expected = pre_group["group_expected"].sum() - pre_expected
        post_peer_actual = post_group["group_actual"].sum() - post_actual
        post_peer_expected = post_group["group_expected"].sum() - post_expected
        pre_peer_eff = safe_divide(pre_peer_actual, pre_peer_expected)
        post_peer_eff = safe_divide(post_peer_actual, post_peer_expected)
        peer_change = post_peer_eff - pre_peer_eff if not pd.isna(pre_peer_eff) and not pd.isna(post_peer_eff) else np.nan
        manager_effect = hotel_change - peer_change if not pd.isna(hotel_change) and not pd.isna(peer_change) else np.nan

        if not included_in_effect:
            manager_effect_to_store = np.nan
            hotel_change_to_store = np.nan
            peer_change_to_store = np.nan
        else:
            manager_effect_to_store = manager_effect
            hotel_change_to_store = hotel_change
            peer_change_to_store = peer_change

        evaluation_note = ""
        if has_overlap:
            evaluation_note = "В окне есть другие Action Logs этого же отеля"
            if overlap_subjects:
                evaluation_note += f": {overlap_subjects}"

        rows.append({
            "action_id": action_id,
            "hotel_id": hotel_id,
            "segment_key": segment_key,
            "action_date": action["action_date"],
            "action_month": action_month,
            "effective_month": effective_month,
            "before_month": before_month,
            "after_month": after_month,
            "subject": action["subject"],
            "outcome": action["outcome"],
            "outcome_group": action.get("outcome_group", ""),
            "effect_type": effect_type,
            "effect_type_label": effect_type_label,
            "pre_window_months": 1,
            "lag_months": np.nan,
            "post_window_months": 1,
            "pre_months_available": pre_months_available,
            "post_months_available": post_months_available,
            "has_overlap": has_overlap,
            "overlapping_actions_count": overlapping_actions_count,
            "included_in_effect": included_in_effect,
            "exclusion_reason": "; ".join(dict.fromkeys(exclusion_reasons)),
            "evaluation_note": evaluation_note,
            "pre_actual": pre_actual,
            "pre_expected": pre_expected,
            "post_actual": post_actual,
            "post_expected": post_expected,
            "pre_efficiency": pre_eff,
            "post_efficiency": post_eff,
            "hotel_change_pp": hotel_change_to_store * 100 if not pd.isna(hotel_change_to_store) else np.nan,
            "peer_change_pp": peer_change_to_store * 100 if not pd.isna(peer_change_to_store) else np.nan,
            "manager_effect_pp": manager_effect_to_store * 100 if not pd.isna(manager_effect_to_store) else np.nan,
            "pre_peer_efficiency": pre_peer_eff,
            "post_peer_efficiency": post_peer_eff,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.replace([np.inf, -np.inf], np.nan)
    result["verdict"] = result.apply(classify_action, axis=1)
    result.loc[result["included_in_effect"] == False, "verdict"] = (
        "Исключено из расчета: " + result.loc[result["included_in_effect"] == False, "exclusion_reason"].fillna("")
    )
    return result

def calculate_worked_vs_not_worked(period_classification, actions_df):
    df = period_classification.copy()
    if actions_df.empty:
        df["has_any_action"] = False
        df["worked_group"] = "Без Action Logs"
    else:
        worked_hotels = set(actions_df["hotel_id"].astype(str).unique().tolist())
        df["has_any_action"] = df["hotel_id"].astype(str).isin(worked_hotels)
        df["worked_group"] = np.where(df["has_any_action"], "Есть Action Logs", "Без Action Logs")
    result = (
        df.groupby("worked_group", as_index=False)
        .agg(
            rows_count=("hotel_id", "count"),
            hotels_count=("hotel_id", "nunique"),
            avg_individual_effect_pp=("individual_effect_pp", "mean"),
            median_individual_effect_pp=("individual_effect_pp", "median"),
            avg_hotel_efficiency=("hotel_efficiency", "mean")
        )
    )
    return result


def calculate_action_count_correlation(period_classification, actions_df, grain):
    df = period_classification.copy()
    if actions_df.empty:
        df["actions_in_period"] = 0
    else:
        act = actions_df.copy()
        act["period"] = make_period_column(act["action_date"], grain)
        counts = (
            act.groupby(["hotel_id", "period"], as_index=False)
            .size().rename(columns={"size": "actions_in_period"})
        )
        df = df.merge(counts, on=["hotel_id", "period"], how="left")
        df["actions_in_period"] = df["actions_in_period"].fillna(0).astype(int)
    clean = df[["hotel_id", "period_label", "actions_in_period", "individual_effect_pp", "hotel_efficiency"]].copy()
    clean = clean.dropna(subset=["individual_effect_pp", "hotel_efficiency"])
    if len(clean) >= 3 and clean["actions_in_period"].nunique() > 1:
        corr_individual = clean["actions_in_period"].corr(clean["individual_effect_pp"])
        corr_efficiency = clean["actions_in_period"].corr(clean["hotel_efficiency"])
    else:
        corr_individual = np.nan
        corr_efficiency = np.nan
    return clean, corr_individual, corr_efficiency


def calculate_outcome_group_period_summary(period_classification, actions_df, grain):
    if actions_df.empty:
        return pd.DataFrame()
    act = actions_df.copy()
    act["period"] = make_period_column(act["action_date"], grain)
    groups = (
        act.groupby(["hotel_id", "period", "outcome_group"], as_index=False)
        .size().rename(columns={"size": "outcome_actions_count"})
    )
    merged = period_classification.merge(
        groups, on=["hotel_id", "period"], how="inner"
    )
    if merged.empty:
        return pd.DataFrame()
    summary = (
        merged.groupby("outcome_group", as_index=False)
        .agg(
            rows_count=("hotel_id", "count"),
            hotels_count=("hotel_id", "nunique"),
            actions_count=("outcome_actions_count", "sum"),
            avg_individual_effect_pp=("individual_effect_pp", "mean"),
            median_individual_effect_pp=("individual_effect_pp", "median"),
            avg_hotel_efficiency=("hotel_efficiency", "mean")
        )
        .sort_values("avg_individual_effect_pp", ascending=False)
    )
    return summary


def prepare_correlation_dataset(period_classification, actions_df, grain):
    df = period_classification.copy()
    if "actions_count" in df.columns:
        df["actions_in_period"] = df["actions_count"].fillna(0).astype(int)
    elif actions_df.empty:
        df["actions_in_period"] = 0
    else:
        act = actions_df.copy()
        act["period"] = make_period_column(act["action_date"], grain)
        counts = (
            act.groupby(["hotel_id", "period"], as_index=False)
            .size().rename(columns={"size": "actions_in_period"})
        )
        df = df.merge(counts, on=["hotel_id", "period"], how="left")
        df["actions_in_period"] = df["actions_in_period"].fillna(0).astype(int)
    cols = [
        "actions_in_period", "actual_value", "expected_value",
        "hotel_efficiency", "segment_efficiency", "individual_effect_pp"
    ]
    available_cols = [c for c in cols if c in df.columns]
    return df[available_cols + ["hotel_id", "period_label"]].copy()
