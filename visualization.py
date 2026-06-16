import numpy as np
import pandas as pd
import plotly.express as px

from config import METRICS, MONTH_ORDER
from utils import safe_divide, make_period_column, make_period_label
from model import calculate_general_seasonality


def create_monthly_actual_expected_fig(enriched_df, metric_name, grain="Месяц"):
    df = enriched_df.copy()
    df["period"] = make_period_column(df["month"], grain)
    df["period_label"] = make_period_label(df["month"], grain)
    period_line = (
        df.groupby(["period", "period_label"], as_index=False)
        .agg(
            actual=(metric_name, "sum"),
            expected=("expected_value", "sum")
        )
        .sort_values("period")
    )
    period_long = period_line.melt(
        id_vars=["period", "period_label"],
        value_vars=["actual", "expected"],
        var_name="type",
        value_name="value"
    )
    period_long["type"] = period_long["type"].replace({
        "actual": "Факт",
        "expected": "Ожидание с учётом сезонности"
    })
    fig = px.line(
        period_long,
        x="period", y="value", color="type",
        markers=True,
        labels={"period": "Период", "value": METRICS[metric_name], "type": ""},
        hover_data={"period_label": True, "period": False},
        title="Факт против сезонного ожидания"
    )
    return fig


def create_general_seasonality_fig(filtered_monthly, metric_name):
    seasonality = calculate_general_seasonality(filtered_monthly, metric_name)
    fig = px.line(
        seasonality,
        x="month_name", y="seasonality_index",
        markers=True,
        labels={"month_name": "Месяц", "seasonality_index": "Сезонный индекс"},
        title="Годичная сезонность выбранной выборки"
    )
    fig.update_layout(xaxis={"categoryorder": "array", "categoryarray": MONTH_ORDER})
    return fig, seasonality


def create_all_metrics_seasonality_fig(filtered_monthly):
    rows = []
    for metric_name, metric_label in METRICS.items():
        seasonality = calculate_general_seasonality(filtered_monthly, metric_name)
        for _, row in seasonality.iterrows():
            rows.append({
                "month_num": row["month_num"],
                "month_name": row["month_name"],
                "metric": metric_label,
                "metric_key": metric_name,
                "metric_value": row[metric_name],
                "seasonality_index": row["seasonality_index"]
            })
    table = pd.DataFrame(rows)
    fig = px.line(
        table,
        x="month_name", y="seasonality_index", color="metric",
        markers=True,
        labels={
            "month_name": "Месяц",
            "seasonality_index": "Сезонный индекс",
            "metric": "Метрика"
        },
        title="Сезонность по всем основным метрикам"
    )
    fig.update_layout(xaxis={"categoryorder": "array", "categoryarray": MONTH_ORDER})
    return fig, table


def create_multi_metric_index_fig(filtered_monthly, metric_names=None, grain="Месяц"):
    if metric_names is None:
        metric_names = list(METRICS.keys())
    metric_names = [metric_name for metric_name in metric_names if metric_name in METRICS]
    df = filtered_monthly.copy()
    df["period"] = make_period_column(df["month"], grain)
    df["period_label"] = make_period_label(df["month"], grain)
    period_data = (
        df.groupby(["period", "period_label"], as_index=False)
        [metric_names].sum()
        .sort_values("period")
    )
    rows = []
    for metric_name in metric_names:
        metric_label = METRICS[metric_name]
        avg = period_data[metric_name].mean()
        for _, row in period_data.iterrows():
            value = safe_divide(row[metric_name], avg)
            rows.append({
                "period": row["period"],
                "period_label": row["period_label"],
                "metric": metric_label,
                "index": value
            })
    long = pd.DataFrame(rows)
    fig = px.line(
        long, x="period", y="index", color="metric",
        markers=True,
        labels={
            "period": "Период", "index": "Индекс к среднему периоду",
            "metric": "Метрика"
        },
        hover_data={"period_label": True, "period": False},
        title="Нормализованная динамика выбранных метрик"
    )
    return fig


def create_verdict_fig(period_classification):
    verdict_counts = (
        period_classification.groupby("verdict", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )
    fig = px.bar(
        verdict_counts,
        x="verdict", y="count",
        labels={"verdict": "Вердикт", "count": "Кол-во строк объект-период"},
        title="Классификация динамики объектов"
    )
    return fig, verdict_counts


def create_worked_fig(worked_summary):
    fig = px.bar(
        worked_summary,
        x="worked_group", y="avg_individual_effect_pp",
        text="hotels_count",
        labels={
            "worked_group": "Группа",
            "avg_individual_effect_pp": "Средний индивидуальный эффект, п.п."
        },
        title="Отели с Action Logs против отелей без Action Logs"
    )
    return fig


def create_action_outcome_fig(action_impact):
    if action_impact.empty:
        return None, pd.DataFrame()
    clean = action_impact.dropna(subset=["manager_effect_pp"]).copy()
    if clean.empty:
        return None, pd.DataFrame()
    by_outcome = (
        clean.groupby("outcome", as_index=False)
        .agg(
            avg_manager_effect_pp=("manager_effect_pp", "mean"),
            median_manager_effect_pp=("manager_effect_pp", "median"),
            actions_count=("manager_effect_pp", "count")
        )
        .sort_values("avg_manager_effect_pp", ascending=False)
    )
    fig = px.bar(
        by_outcome,
        x="outcome", y="avg_manager_effect_pp",
        text="actions_count",
        labels={"outcome": "Outcome", "avg_manager_effect_pp": "Средний эффект, п.п."},
        title="Эффект Action Logs по outcome"
    )
    return fig, by_outcome


def create_correlation_heatmap(correlation_df):
    rename_map = {
        "actions_in_period": "Action Logs в периоде",
        "actual_value": "Факт",
        "expected_value": "Ожидание",
        "hotel_efficiency": "Эффективность отеля",
        "segment_efficiency": "Фон похожих объектов",
        "individual_effect_pp": "Индивидуальный эффект, п.п."
    }
    numeric = correlation_df.select_dtypes(include=[np.number]).copy()
    if numeric.empty or len(numeric) < 3:
        return None, pd.DataFrame()
    numeric = numeric.rename(columns=rename_map)
    numeric = numeric.loc[:, ~numeric.columns.duplicated()].copy()
    if numeric.shape[1] < 2:
        return None, pd.DataFrame()
    corr = numeric.corr()
    corr = corr.loc[:, ~corr.columns.duplicated()].copy()
    corr = corr.loc[~corr.index.duplicated(), :].copy()
    fig = px.imshow(
        corr,
        text_auto=".2f", aspect="auto",
        zmin=-1, zmax=1,
        color_continuous_scale=["#59758C", "#FFFFFF", "#CC7D5E"],
        labels={"color": "Корреляция"},
        title="Матрица корреляций по объект-периодам"
    )
    fig.update_layout(height=560, margin=dict(l=20, r=20, t=70, b=20))
    fig.update_traces(hovertemplate="%{x}<br>%{y}<br>Корреляция: %{z:.2f}<extra></extra>")
    return fig, corr


def create_action_subject_outcome_heatmap(action_impact, min_actions_for_cell=3):
    if action_impact.empty or "manager_effect_pp" not in action_impact.columns:
        return None, pd.DataFrame()
    clean = action_impact.dropna(subset=["manager_effect_pp"]).copy()
    if clean.empty:
        return None, pd.DataFrame()
    grouped = (
        clean.groupby(["subject", "outcome"], as_index=False)
        .agg(
            avg_manager_effect_pp=("manager_effect_pp", "mean"),
            actions_count=("manager_effect_pp", "count")
        )
    )
    grouped = grouped[grouped["actions_count"] >= min_actions_for_cell]
    if grouped.empty:
        return None, grouped
    pivot = grouped.pivot(index="subject", columns="outcome", values="avg_manager_effect_pp").fillna(0)
    fig = px.imshow(
        pivot, text_auto=".1f", aspect="auto",
        color_continuous_scale=["#59758C", "#FFFFFF", "#CC7D5E"],
        labels={"color": "Manager effect, п.п."},
        title="Тепловая матрица: средний эффект по Subject × Outcome"
    )
    fig.update_layout(
        height=max(520, min(1100, 28 * len(pivot.index) + 180)),
        margin=dict(l=20, r=20, t=70, b=20)
    )
    return fig, grouped


def create_ttest_bar_fig(ttest_table):
    if ttest_table.empty:
        return None
    df = ttest_table.copy()
    df = df.dropna(subset=["Разница средних"])
    if df.empty:
        return None
    fig = px.bar(
        df,
        y="Проверка", x="Разница средних",
        color="p-value",
        orientation="h",
        color_continuous_scale=["#F4E5DF", "#CC7D5E", "#59758C"],
        labels={"Разница средних": "Разница средних, п.п.", "p-value": "p-value"},
        title="T-статистика: сила и надёжность различий"
    )
    fig.update_layout(
        height=max(420, 70 * len(df) + 160),
        yaxis={"categoryorder": "total ascending"}
    )
    fig.add_vline(x=0, line_width=1, line_dash="dash")
    return fig
