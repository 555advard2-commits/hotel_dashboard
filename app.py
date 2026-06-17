import html
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    METRICS, BOOKING_COLUMNS, ACTION_COLUMNS,
    GLOSSARY, METHODOLOGY_LINKS, SUBJECT_DESCRIPTIONS,
    MONTH_ORDER, MONTH_NAMES
)
from utils import (
    safe_divide, format_number, to_csv_bytes, normalize_columns,
    month_diff_inclusive
)
from ui_components import (
    inject_global_css, apply_corporate_plotly_theme,
    corporate_card, metric_status, show_glossary, safe_plotly_chart
)
from data import (
    read_csv_auto, check_columns, prepare_bookings_data, prepare_actions_data
)
from model import (
    add_expected_values, calculate_hotel_scores, calculate_metric_summary,
    calculate_seasonality_improved
)
from analysis import (
    evaluate_sample_quality, calculate_period_classification,
    calculate_action_impact
)
from visualization import (
    create_monthly_actual_expected_fig, create_all_metrics_seasonality_fig,
    create_multi_metric_index_fig,
    create_action_outcome_fig
)
from statistics import (
    build_manager_action_ttest, format_pvalue
)
from action_log_effect_calculator import (
    calculate_action_log_effects, prepare_monthly_metrics, validate_effect_config
)
from action_log_effect_ui import render_action_log_effect_ui
from hotel_clustering import (
    build_monthly_panel, build_hotel_features,
    run_seasonal_clustering, run_economic_clustering, run_complex_clustering,
    build_cluster_profile, attach_bizdev_actions, build_bizdev_effect_by_cluster
)


@contextmanager
def timed_step(step_name):
    start_time = time.perf_counter()
    print(f"[PERF] START {step_name}", flush=True)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        print(f"[PERF] END   {step_name}: {elapsed:.3f}s", flush=True)


print(f"[PERF] ===== Streamlit rerun {time.strftime('%Y-%m-%d %H:%M:%S')} =====", flush=True)


CLUSTER_FIELD_DESCRIPTIONS = {
    "cluster_id": "Технический номер кластера. Используется для связи таблиц и графиков.",
    "cluster_name": "Бизнес-название группы, сформированное по профилю отелей внутри кластера.",
    "cluster_explanation": "Краткое объяснение, какие признаки сильнее всего отличают кластер от общей выборки.",
    "is_STR": "Признак апартаментов. True — апартаменты, False — отель.",
    "hotels_count": "Количество отелей в кластере.",
    "str_share": "Доля апартаментов внутри кластера.",
    "avg_monthly_revenue": "Средний месячный доход объекта за выбранный период.",
    "avg_revenue": "Средний месячный доход отелей внутри кластера.",
    "median_revenue": "Медианный месячный доход: типичный объект кластера без сильного влияния выбросов.",
    "avg_monthly_roomnights": "Среднее количество ночей в месяц.",
    "avg_roomnights": "Среднее количество ночей в месяц по кластеру.",
    "avg_monthly_gbb": "Среднее количество бронирований в месяц.",
    "avg_gbb": "Среднее количество бронирований в месяц по кластеру.",
    "ADR": "Average Daily Rate: revenue / roomnights. Показывает средний доход на одну ночь.",
    "LOS": "Length of Stay: roomnights / GBB. Показывает среднюю длительность проживания.",
    "SI": "Seasonality Index. Чем выше, тем сильнее пиковый месяц отличается от обычного месяца.",
    "CV": "Coefficient of Variation. Чем выше, тем менее стабильна динамика по месяцам.",
    "summer_share": "Доля спроса, приходящаяся на июнь, июль и август.",
    "winter_share": "Доля спроса, приходящаяся на декабрь, январь и февраль.",
    "growth_2025_vs_2024": "Рост 2025 к 2024 по revenue. 0.10 означает примерно +10%.",
    "months_available": "Количество месяцев, по которым у объекта есть данные в выбранной выборке.",
    "top_demand_months": "Три календарных месяца с максимальной средней долей спроса.",
    "hotels_with_actions": "Количество отелей кластера, у которых были Action Logs после фильтров.",
    "actions_count": "Количество действий BizDev в кластере после фильтров.",
    "avg_effect": "Средний manager effect по действиям в кластере.",
    "median_effect": "Медианный manager effect: более устойчив к выбросам, чем среднее.",
    "success_rate": "Доля действий с положительным manager effect.",
}


def render_cluster_explanations(title, columns):
    items = []
    for column in columns:
        description = CLUSTER_FIELD_DESCRIPTIONS.get(column)
        if description:
            items.append(f"<li><b>{html.escape(column)}</b> — {html.escape(description)}</li>")
    if not items:
        return
    st.markdown(
        f"""
        <div class="corporate-card">
            <h3>{html.escape(title)}</h3>
            <ul>{''.join(items)}</ul>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_method_note(title, body):
    st.markdown(
        f"""
        <div class="corporate-card">
            <h3>{html.escape(title)}</h3>
            <p>{html.escape(body)}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_cluster_takeaways(title, items):
    clean_items = [str(item).strip() for item in items if str(item).strip()]
    if not clean_items:
        return
    list_html = "".join(f"<li>{html.escape(item)}</li>" for item in clean_items)
    st.markdown(
        f"""
        <div class="corporate-card">
            <h3>{html.escape(title)}</h3>
            <ul>{list_html}</ul>
        </div>
        """,
        unsafe_allow_html=True
    )


def build_cluster_quality_takeaways(cluster_profile, cluster_metrics, requested_k):
    if cluster_profile is None or cluster_profile.empty:
        return ["Профиль кластеров пустой: выводы по качеству недоступны."]
    hotels_total = int(cluster_profile["hotels_count"].sum())
    actual_k = int(cluster_profile["cluster_id"].nunique())
    min_size = int(cluster_profile["hotels_count"].min())
    max_size = int(cluster_profile["hotels_count"].max())
    max_share = max_size / hotels_total if hotels_total else 0
    silhouette = cluster_metrics.get("silhouette_score")
    items = [
        f"Запрошено k = {requested_k}, фактически построено {actual_k} кластеров на {hotels_total} отелях.",
        f"Размеры кластеров: минимум {min_size}, максимум {max_size}; крупнейший кластер забирает {max_share:.0%} выборки.",
    ]
    if pd.notna(silhouette):
        if silhouette >= 0.35:
            items.append(f"Silhouette score = {silhouette:.3f}: разделение выглядит достаточно читаемым.")
        elif silhouette >= 0.15:
            items.append(f"Silhouette score = {silhouette:.3f}: кластеры есть, но границы умеренно размыты.")
        else:
            items.append(f"Silhouette score = {silhouette:.3f}: кластеры близки друг к другу, k и набор признаков стоит перепроверить.")
    if min_size < 20:
        items.append(f"Минимальный кластер содержит {min_size} отелей: для бизнес-выводов это мало, лучше проверить меньшее k.")
    if max_share >= 0.70 and actual_k > 1:
        items.append("Один кластер забирает больше 70% выборки: типология может быть перекошена в сторону одной большой группы.")
    return items


def build_cluster_profile_takeaways(cluster_profile):
    if cluster_profile is None or cluster_profile.empty:
        return []
    items = []
    for _, row in cluster_profile.sort_values("hotels_count", ascending=False).iterrows():
        name = row.get("cluster_name", f"Кластер {row.get('cluster_id', '')}")
        count = int(row.get("hotels_count", 0))
        str_share = row.get("str_share")
        explanation = row.get("cluster_explanation", "")
        str_part = f", STR {str_share:.0%}" if pd.notna(str_share) else ""
        items.append(f"{name}: {count} отелей{str_part}. Отличия: {explanation}.")
    return items


def build_seasonality_takeaways(seasonality_df):
    if seasonality_df is None or seasonality_df.empty:
        return []
    df = seasonality_df.copy()
    df = df[pd.to_numeric(df["seasonality_index"], errors="coerce").notna()].copy()
    if df.empty:
        return ["Сезонный индекс не рассчитался: в выбранной базе недостаточно валидных месячных значений."]
    peak = df.loc[df["seasonality_index"].idxmax()]
    low = df.loc[df["seasonality_index"].idxmin()]
    spread = (
        df.groupby("cluster_name")["seasonality_index"]
        .agg(lambda x: float(x.max() - x.min()))
        .sort_values(ascending=False)
    )
    most_seasonal = spread.index[0] if not spread.empty else None
    items = [
        f"Максимальный сезонный пик: {peak['cluster_name']} в месяце {peak['month_name']} — индекс {peak['seasonality_index']:.2f}.",
        f"Самая слабая сезонная точка: {low['cluster_name']} в месяце {low['month_name']} — индекс {low['seasonality_index']:.2f}.",
    ]
    if most_seasonal is not None:
        items.append(f"Самый выраженный сезонный профиль у кластера «{most_seasonal}»: у него максимальный размах индекса по месяцам.")
    return items


def build_heatmap_takeaways(heat_z):
    if heat_z is None or heat_z.empty:
        return []
    items = []
    for cluster_name, row in heat_z.fillna(0).iterrows():
        top_high = row.sort_values(ascending=False).head(2)
        top_low = row.sort_values(ascending=True).head(2)
        high_text = ", ".join(f"{col} ({value:.1f})" for col, value in top_high.items())
        low_text = ", ".join(f"{col} ({value:.1f})" for col, value in top_low.items())
        items.append(f"{cluster_name}: выше среднего — {high_text}; ниже среднего — {low_text}.")
    return items


st.set_page_config(
    page_title="Hotel Seasonality & Manager Impact",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏨"
)


@st.cache_data(show_spinner=False)
def build_clustering_features_cached(monthly_df):
    with timed_step("clustering.build_monthly_panel"):
        monthly_panel = build_monthly_panel(monthly_df)
    with timed_step("clustering.build_hotel_features"):
        return build_hotel_features(monthly_panel)


@st.cache_data(show_spinner=False)
def run_clustering_cached(features_df, cluster_mode_value, cluster_k_value):
    with timed_step("clustering.run_algorithm"):
        if cluster_mode_value == "Сезонная":
            return run_seasonal_clustering(features_df, cluster_k_value)
        if cluster_mode_value == "Экономическая":
            return run_economic_clustering(features_df, cluster_k_value)
        return run_complex_clustering(features_df, cluster_k_value)


@st.cache_data(show_spinner=False)
def evaluate_sample_quality_cached(
    monthly_data, selected_metrics, window_start_value, window_end_value,
    require_full_observed_span_value, min_nonzero_months_value,
    max_zero_months_value, max_zero_share_pct_value, min_observed_months_value
):
    return evaluate_sample_quality(
        monthly_data,
        selected_metrics=selected_metrics,
        window_start=window_start_value,
        window_end=window_end_value,
        require_full_observed_span=require_full_observed_span_value,
        min_nonzero_months=min_nonzero_months_value,
        max_zero_months=max_zero_months_value,
        max_zero_share_pct=max_zero_share_pct_value,
        min_observed_months=min_observed_months_value,
        strict_all_selected_metrics=True,
    )


@st.cache_data(show_spinner=False)
def build_model_outputs_cached(monthly_data, actions_data, metric_name, grain, threshold_value):
    enriched = add_expected_values(monthly_data, metric_name)
    scores = calculate_hotel_scores(enriched, metric_name)
    summary = calculate_metric_summary(monthly_data)
    period_rows = calculate_period_classification(enriched, actions_data, metric_name, grain, threshold_value)
    return enriched, scores, summary, period_rows


@st.cache_data(show_spinner=False)
def calculate_action_impact_cached(
    actions_data, enriched_data, metric_name, pre_months, post_months,
    lag_months_value, min_months, overlap_policy_value
):
    return calculate_action_impact(
        actions_data, enriched_data, metric_name,
        pre_months, post_months, lag_months_value,
        min_months, overlap_policy_value,
    )


@st.cache_data(show_spinner=False)
def build_ttest_outputs_cached(monthly_data, action_rows, alpha_value, min_observations_value, dedup_enabled_value):
    return build_manager_action_ttest(
        monthly_data,
        action_rows,
        alpha=alpha_value,
        min_observations=min_observations_value,
        dedup_enabled=dedup_enabled_value,
    )


@st.cache_data(show_spinner=False)
def calculate_automatic_action_effects_cached(actions_data, monthly_data, metric_name):
    monthly_metrics = prepare_monthly_metrics(monthly_data)
    effects = calculate_action_log_effects(actions_data, monthly_metrics, metric_name)
    return monthly_metrics, effects


inject_global_css()
apply_corporate_plotly_theme()
try:
    validate_effect_config()
except ValueError as error:
    st.error(str(error))
    st.stop()

st.markdown(
    """
    <div class="corporate-hero">
        <h1>Hotel Seasonality & Manager Impact Dashboard</h1>
        <p>Одна локалия · сезонность · Action Logs · оценка индивидуального эффекта менеджеров</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption(
    "Одна локалия. Без регионального разделения. Цель – понять, где рост объясняется сезонностью, "
    "а где объект растёт сильнее локального фона после работы менеджеров."
)

st.sidebar.header("1. Загрузка данных")

bookings_file = st.sidebar.file_uploader(
    "Данные по отелям",
    type=["csv"],
    help="Первая таблица: hotel_id, booking_created_date, is_STR, gbb, roomnights, sales_volumes_rub, revenue_rub.",
    key="bookings_csv_upload"
)
actions_file = st.sidebar.file_uploader(
    "Action Logs",
    type=["csv"],
    help="Вторая таблица: action_date, subject, outcome, hotel_id.",
    key="actions_csv_upload"
)


def normalize_uploaded_file(uploaded):
    if isinstance(uploaded, (list, tuple)):
        return uploaded[0] if uploaded else None
    return uploaded


bookings_file = normalize_uploaded_file(bookings_file)
actions_file = normalize_uploaded_file(actions_file)

bookings_raw = None
actions_raw = None

if bookings_file is not None:
    with timed_step("bookings.read_csv"):
        bookings_raw = normalize_columns(read_csv_auto(bookings_file))
    bookings_size_mb = getattr(bookings_file, "size", 0) / (1024 * 1024)
    st.sidebar.markdown(
        f"""
        <div class="upload-status">
            <div class="upload-status-badge">Загружен</div>
            <div class="upload-status-title">Данные по отелям</div>
            <div class="upload-status-body">{html.escape(bookings_file.name)}</div>
            <div class="upload-status-body">Размер: {bookings_size_mb:.1f} МБ · строк: {len(bookings_raw)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        """
        <div class="upload-status upload-status-empty">
            <div class="upload-status-title">Данные по отелям</div>
            <div class="upload-status-body">Файл не выбран</div>
        </div>
        """,
        unsafe_allow_html=True
    )

if actions_file is not None:
    with timed_step("actions.read_csv"):
        actions_raw = normalize_columns(read_csv_auto(actions_file))
    actions_size_mb = getattr(actions_file, "size", 0) / (1024 * 1024)
    st.sidebar.markdown(
        f"""
        <div class="upload-status">
            <div class="upload-status-badge">Загружен</div>
            <div class="upload-status-title">Action Logs</div>
            <div class="upload-status-body">{html.escape(actions_file.name)}</div>
            <div class="upload-status-body">Размер: {actions_size_mb:.1f} МБ · строк: {len(actions_raw)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        """
        <div class="upload-status upload-status-empty">
            <div class="upload-status-title">Action Logs</div>
            <div class="upload-status-body">Файл не выбран</div>
        </div>
        """,
        unsafe_allow_html=True
    )

if bookings_file is None or actions_file is None:
    missing_files = []
    if bookings_file is None:
        missing_files.append("таблицу продаж")
    if actions_file is None:
        missing_files.append("таблицу Action Logs")
    st.info(f"Загрузи CSV: {', '.join(missing_files)}.")
    st.stop()

with timed_step("validation.check_columns"):
    check_columns(bookings_raw, BOOKING_COLUMNS, "продаж")
    check_columns(actions_raw, ACTION_COLUMNS, "Action Logs")

with timed_step("bookings.prepare_data"):
    bookings_df, monthly_df, hotel_attrs = prepare_bookings_data(bookings_raw)
with timed_step("actions.prepare_data"):
    actions_df = prepare_actions_data(actions_raw)

st.sidebar.header("2. Подготовка выборки")

available_months = sorted(monthly_df["month"].dropna().unique())
if len(available_months) == 0:
    st.error("В таблице продаж не найдено ни одного месяца с корректной датой.")
    st.stop()
available_months = [pd.Timestamp(x) for x in available_months]

window_start = st.sidebar.selectbox(
    "Начало периода",
    available_months, index=0,
    format_func=lambda x: f"{MONTH_NAMES.get(pd.Timestamp(x).month, '')} {pd.Timestamp(x).year}",
    help="Первый месяц, который войдёт в анализ."
)

window_end = st.sidebar.selectbox(
    "Конец периода",
    available_months, index=len(available_months) - 1,
    format_func=lambda x: f"{MONTH_NAMES.get(pd.Timestamp(x).month, '')} {pd.Timestamp(x).year}",
    help="Последний месяц, который войдёт в анализ."
)

object_type = st.sidebar.selectbox(
    "Тип объекта",
    ["Все", "Апартаменты", "Отели"],
    index=0,
    help="Все — все объекты; Апартаменты — только апартаменты; Отели — только отели."
)

require_full_observed_span = st.sidebar.checkbox(
    "Только отели с данными за весь период",
    value=False,
    help="Оставляет только те отели, у которых есть данные за каждый месяц выбранного периода."
)

use_zero_share_filter = st.sidebar.checkbox(
    "Ограничивать долю нулевых месяцев по GBB",
    value=True,
    help="Если выключено, отели не исключаются из-за месяцев с GBB = 0."
)

window_months = month_diff_inclusive(window_start, window_end)

if use_zero_share_filter:
    max_zero_share_pct = st.sidebar.slider(
        "Максимальная доля нулевых месяцев по GBB",
        min_value=0, max_value=50, value=20,
        help="Фильтр ограничивает долю месяцев внутри выбранного периода, где у отеля значение GBB равно 0."
    )
else:
    max_zero_share_pct = 100

# Derived quality params for sample-quality filtering
sample_metrics = ["gbb"]
quality_preset = "Пользовательская"
strict_all_selected_metrics = True
min_observed_months = max(1, window_months if require_full_observed_span else 1)
max_zero_months = int(np.ceil(window_months * max_zero_share_pct / 100))
min_nonzero_months = max(0, window_months - max_zero_months)
zero_filter_status = (
    f"включён, максимум {max_zero_share_pct}%"
    if use_zero_share_filter
    else "выключен"
)

# Отели до фильтров (для статуса M)
total_before_filter = monthly_df[
    (monthly_df["month"] >= window_start) & (monthly_df["month"] <= window_end)
]["hotel_id"].nunique()

with timed_step("block2.evaluate_sample_quality"):
    sample_quality, prepared_monthly_strict = evaluate_sample_quality_cached(
        monthly_df,
        tuple(sample_metrics),
        window_start,
        window_end,
        require_full_observed_span,
        min_nonzero_months,
        max_zero_months,
        max_zero_share_pct,
        min_observed_months,
    )

base_monthly_for_model = prepared_monthly_strict.copy()

# Карта object_type → hotel_type_filter для экспорта в HTML-отчёт
hotel_type_filter = {"Все": "Все", "Апартаменты": "Апартаменты", "Отели": "Отели"}[object_type]

if object_type == "Апартаменты":
    base_monthly_for_model = base_monthly_for_model[base_monthly_for_model["is_STR"] == True]
elif object_type == "Отели":
    base_monthly_for_model = base_monthly_for_model[base_monthly_for_model["is_STR"] == False]

total_after_filter = base_monthly_for_model["hotel_id"].nunique()
pct_saved = (total_after_filter / total_before_filter * 100) if total_before_filter > 0 else 0
total_rows = len(base_monthly_for_model)

st.sidebar.markdown(
    f"**После фильтров:**  \n"
    f"отелей: {total_after_filter} из {total_before_filter}  \n"
    f"сохранено: {pct_saved:.0f}%  \n"
    f"месяцев × отелей: {total_rows}"
)

if base_monthly_for_model.empty:
    st.error(
        "После подготовки выборки не осталось данных. Ослабь фильтры: увеличь допустимую долю нулей, "
        "отключи фильтр полного покрытия или выбери более широкий период."
    )
    if not sample_quality.empty:
        st.dataframe(sample_quality.head(100), use_container_width=True, hide_index=True)
    st.stop()

st.sidebar.header("3. Метрики оценки результата")

metric = st.sidebar.selectbox(
    "Главная метрика",
    list(METRICS.keys()),
    format_func=lambda x: METRICS[x],
    index=0,
    help="По этой метрике система сравнивает ожидаемый и фактический результат отеля и считает эффект действий."
)

additional_metric_options = [metric_name for metric_name in METRICS.keys() if metric_name != metric]
additional_metrics = st.sidebar.multiselect(
    "Дополнительные метрики",
    additional_metric_options,
    default=[],
    format_func=lambda x: METRICS[x],
    help="Эти метрики показываются для сравнения, но не влияют на главный расчёт эффекта."
)
additional_metrics = [metric_name for metric_name in additional_metrics if metric_name != metric]
display_metrics = [metric] + additional_metrics

dynamics_grain_label = st.sidebar.selectbox(
    "Показывать динамику по",
    ["Месяцам", "Кварталам", "Годам"],
    index=0,
    help="Определяет, как группировать графики динамики: по месяцам, кварталам или годам."
)
graining = {
    "Месяцам": "Месяц",
    "Кварталам": "Квартал",
    "Годам": "Год"
}[dynamics_grain_label]

with timed_step("block3.prepare_metric_sample"):
    filtered_monthly = base_monthly_for_model.copy()

with timed_step("block4.link_actions_to_sample"):
    sample_hotels_for_actions = set(filtered_monthly["hotel_id"].astype(str).unique())
    actions_after_sample_filters = actions_df[
        actions_df["hotel_id"].astype(str).isin(sample_hotels_for_actions)
        & (actions_df["action_month"] >= window_start)
        & (actions_df["action_month"] <= window_end)
    ].copy()

st.sidebar.header("4. Action Logs")

impact_outcomes = ["Published", "Fixed", "Returned"]
selected_outcomes = st.sidebar.multiselect(
    "Outcomes для оценки влияния",
    impact_outcomes,
    default=impact_outcomes,
    help="Фильтр определяет, какие результаты действий из Action Logs учитывать в анализе."
)

all_subjects = sorted(actions_df["subject"].dropna().unique().tolist())
selected_subjects = st.sidebar.multiselect(
    "Subjects",
    all_subjects,
    default=all_subjects,
    help="Тематики действий из Action Logs. По умолчанию выбраны все темы; ненужные можно убрать."
)

with timed_step("block4.apply_outcome_subject_filters"):
    filtered_actions = actions_after_sample_filters[
        actions_after_sample_filters["outcome"].isin(selected_outcomes)
        & actions_after_sample_filters["subject"].isin(selected_subjects)
    ].copy()

with timed_step("action_logs.automatic_effects_cached"):
    automatic_monthly_metrics, automatic_action_effects = calculate_automatic_action_effects_cached(
        filtered_actions,
        filtered_monthly,
        metric,
    )

actions_before_block4 = len(actions_after_sample_filters)
actions_after_block4 = len(filtered_actions)
hotels_with_actions_after_block4 = filtered_actions["hotel_id"].nunique()

st.sidebar.markdown(
    f"**После фильтров:**  \n"
    f"Action Logs: {actions_after_block4} из {actions_before_block4}  \n"
    f"отелей с Action Logs: {hotels_with_actions_after_block4}"
)

pre_window_months = 3
lag_months = 0
post_window_months = 3
min_months_per_side = 2
overlap_policy = "Помечать наложения, но не исключать"
threshold_percent = 10
threshold = threshold_percent / 100
hotel_search = ""

if hotel_search != "":
    filtered_monthly = filtered_monthly[
        filtered_monthly["hotel_id"].str.contains(hotel_search, case=False, na=False)
    ]

hotels_after_filter = filtered_monthly["hotel_id"].unique().tolist()
filtered_actions = filtered_actions[filtered_actions["hotel_id"].isin(hotels_after_filter)]

if filtered_monthly.empty:
    st.warning("После фильтров не осталось данных по отелям.")
    st.stop()

# Основные расчёты
with timed_step("model.outputs_cached"):
    enriched_df, hotel_scores, metric_summary, period_classification = build_model_outputs_cached(
        filtered_monthly,
        filtered_actions,
        metric,
        graining,
        threshold,
    )
with timed_step("analysis.action_impact_cached"):
    action_impact = calculate_action_impact_cached(
        filtered_actions,
        enriched_df,
        metric,
        pre_window_months,
        post_window_months,
        lag_months,
        min_months_per_side,
        overlap_policy,
    )
with timed_step("statistics.ttest_outputs_cached"):
    ttest_table, ttest_pairs, ttest_diagnostics, ttest_run_summary = build_ttest_outputs_cached(
        filtered_monthly,
        filtered_actions,
        0.05,
        20,
        True,
    )

# Информация о текущей генеральной совокупности
source_hotels = monthly_df["hotel_id"].nunique()
source_monthly_rows = len(monthly_df)
source_actions = len(actions_df)
prepared_hotels = base_monthly_for_model["hotel_id"].nunique()
prepared_monthly_rows = len(base_monthly_for_model)
current_hotels = filtered_monthly["hotel_id"].nunique()
current_monthly_rows = len(filtered_monthly)
current_actions = len(filtered_actions)
current_period_rows = len(period_classification)
current_str_hotels = filtered_monthly[filtered_monthly["is_STR"] == True]["hotel_id"].nunique()
current_hotel_hotels = filtered_monthly[filtered_monthly["is_STR"] == False]["hotel_id"].nunique()

if action_impact.empty:
    calculated_actions_count = 0
    excluded_actions_count = 0
    overlapped_actions_count = 0
else:
    calculated_actions_count = int(action_impact["manager_effect_pp"].notna().sum())
    excluded_actions_count = int((action_impact["included_in_effect"] == False).sum()) if "included_in_effect" in action_impact.columns else 0
    overlapped_actions_count = int((action_impact["has_overlap"] == True).sum()) if "has_overlap" in action_impact.columns else 0

st.info(
    f"Окно анализа: {window_start.strftime('%Y-%m')} → {window_end.strftime('%Y-%m')}. "
    f"После подготовки выборки: {prepared_hotels:,} объектов, {prepared_monthly_rows:,} строк hotel_id × month. "
    f"После UI-фильтров: {current_hotels:,} объектов, "
    f"{current_monthly_rows:,} строк hotel_id × month, {current_actions:,} Action Logs. "
    f"STR: {current_str_hotels:,}, HOTEL: {current_hotel_hotels:,}. "
    f"Эффект рассчитан для {calculated_actions_count:,} Action Logs; "
    f"исключено/не хватило данных: {excluded_actions_count:,}; "
    f"с наложениями: {overlapped_actions_count:,}.".replace(",", " ")
)

with st.expander("Как фильтры и ползунки меняют генеральную совокупность и расчёты", expanded=False):
    st.markdown(
        f"""
        - Исходная база после очистки: **{source_hotels:,} объектов**, **{source_monthly_rows:,} строк hotel_id × month**, **{source_actions:,} Action Logs**.
        - Окно анализа: **{window_start.strftime('%Y-%m')} → {window_end.strftime('%Y-%m')}**, всего **{window_months} мес.**
        - После подготовки качества выборки: **{prepared_hotels:,} объектов**, **{prepared_monthly_rows:,} строк hotel_id × month**.
        - Текущая база после UI-фильтров: **{current_hotels:,} объектов**, **{current_monthly_rows:,} строк hotel_id × month**, **{current_actions:,} Action Logs**.
        - Сейчас эффект удалось рассчитать для **{calculated_actions_count:,} Action Logs**. Остальные не вошли в расчёт эффекта из-за нехватки месяцев до/после, нулевого ожидания или выбранной политики наложений.
        """.replace(",", " ")
    )

# Вкладки
tab2, tab4, tab7, tab8, tab10 = st.tabs([
    "Сезонность",
    "Action Logs",
    "T-статистика",
    "Кластеризация",
    "Методика и глоссарий"
])

with tab2:
    st.subheader("Общая картина")
    total_hotels_overview = enriched_df["hotel_id"].nunique()
    total_actions_overview = len(filtered_actions)
    total_metric_overview = enriched_df[metric].sum()
    total_expected_overview = enriched_df["expected_value"].sum()
    total_eff_overview = safe_divide(total_metric_overview, total_expected_overview)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Отелей", f"{total_hotels_overview:,.0f}".replace(",", " "))
    col2.metric("Action Logs", f"{total_actions_overview:,.0f}".replace(",", " "))
    col3.metric(METRICS[metric], f"{total_metric_overview:,.0f}".replace(",", " "))
    col4.metric("Факт / ожидание", f"{total_eff_overview:.3f}" if not pd.isna(total_eff_overview) else "—")

    st.subheader("Метрики по текущей выборке")
    st.caption(
        "Главный расчёт эффекта строится по главной метрике. Дополнительные метрики выводятся только для сравнения."
    )
    metric_summary_shown = metric_summary[metric_summary["metric"].isin(display_metrics)].copy()
    metric_order = {metric_name: index for index, metric_name in enumerate(display_metrics)}
    metric_summary_shown["metric_order"] = metric_summary_shown["metric"].map(metric_order)
    metric_summary_shown = metric_summary_shown.sort_values("metric_order")
    metric_summary_shown["actual_sum"] = metric_summary_shown["actual_sum"].round(2)
    metric_summary_shown["expected_sum"] = metric_summary_shown["expected_sum"].round(2)
    metric_summary_shown["efficiency"] = metric_summary_shown["efficiency"].round(3)
    metric_summary_shown["deviation_pct"] = (metric_summary_shown["deviation_pct"] * 100).round(2)

    st.markdown(f"Методика: [что значат 4 метрики]({METHODOLOGY_LINKS['metrics']}) · [как считается ожидание]({METHODOLOGY_LINKS['expected']}) · [как считается эффективность]({METHODOLOGY_LINKS['efficiency']})")

    st.markdown("#### Check-list выбранных метрик")
    for _, row in metric_summary_shown.iterrows():
        icon, status_text = metric_status(row["efficiency"])
        st.markdown(
            f"{icon} **{row['metric_label']}** – {status_text}; "
            f"факт: `{format_number(row['actual_sum'], 0)}`, "
            f"ожидание: `{format_number(row['expected_sum'], 0)}`, "
            f"эффективность: `{format_number(row['efficiency'], 3)}`"
        )

    st.dataframe(
        metric_summary_shown[["metric_label", "actual_sum", "expected_sum", "efficiency", "deviation_pct", "description"]],
        use_container_width=True, hide_index=True
    )

    st.subheader("Факт против сезонного ожидания")
    safe_plotly_chart(
        create_monthly_actual_expected_fig(enriched_df, metric, graining),
        use_container_width=True,
        key="plot_monthly_actual_expected_overview"
    )

    st.subheader("Нормализованная динамика выбранных метрик")
    st.caption("Линии приведены к среднему периоду своей метрики. Так можно сравнить форму динамики, даже если единицы измерения разные.")
    safe_plotly_chart(
        create_multi_metric_index_fig(filtered_monthly, display_metrics, graining),
        use_container_width=True,
        key="plot_multi_metric_index"
    )


    st.divider()
    st.subheader("Сезонность по всем основным метрикам")
    fig, seasonality_table = create_all_metrics_seasonality_fig(filtered_monthly)
    safe_plotly_chart(fig, use_container_width=True, key="plot_seasonality_all_metrics")
    st.caption("Индекс 1.20 = месяц на 20% сильнее среднего. Индекс 0.80 = месяц на 20% слабее среднего.")
    seasonality_table["metric_value"] = seasonality_table["metric_value"].round(2)
    seasonality_table["overall_avg"] = seasonality_table["overall_avg"].round(2)
    seasonality_table["seasonality_index"] = seasonality_table["seasonality_index"].round(3)
    st.dataframe(
        seasonality_table[["metric", "month_num", "month_name", "metric_value", "overall_avg", "seasonality_index", "hotel_count"]],
        use_container_width=True,
        hide_index=True
    )
    render_method_note(
        "Проверка формулы",
        "График по четырём основным метрикам считается по новой формуле: среднее значение календарного месяца по отелям делится на взвешенное среднее по всем месяцам; вес месяца — количество отелей, месяцы-выбросы исключаются по Z-score."
    )

    st.subheader("Сезонность по типу объекта")
    st.caption("Регионов нет. Сравнение делается внутри одной локалии по типу объекта: Апартаменты / Отели.")
    object_monthly = filtered_monthly.copy()
    object_monthly["object_type_label"] = np.where(object_monthly["is_STR"] == True, "Апартаменты", "Отели")
    seg_seasonality = calculate_seasonality_improved(
        object_monthly,
        metric,
        group_cols=["object_type_label"],
    )
    seg_seasonality["month_name"] = seg_seasonality["month_num"].map(MONTH_NAMES)
    fig = px.line(
        seg_seasonality,
        x="month_name", y="seasonality_index", color="object_type_label",
        markers=True,
        labels={
            "month_name": "Месяц",
            "seasonality_index": "Сезонный индекс",
            "object_type_label": "Тип объекта",
            "hotel_count": "Отелей в месяце",
        },
        hover_data=["hotel_count"],
    )
    fig.update_layout(xaxis={"categoryorder": "array", "categoryarray": MONTH_ORDER})
    safe_plotly_chart(fig, use_container_width=True, key="plot_seasonality_by_segment")
    shown_object_types = ", ".join(sorted(seg_seasonality["object_type_label"].dropna().unique()))
    render_method_note(
        "Вывод по типу объекта",
        f"На графике показаны типы объекта из текущей выборки Блока 2: {shown_object_types}. Если выбран фильтр «Все», одновременно отображаются Апартаменты и Отели; если выбран один тип объекта, график показывает только его."
    )

with tab4:
    st.subheader("Оценка влияния Action Logs")
    render_action_log_effect_ui(automatic_action_effects, METRICS[metric])
    st.divider()
    st.subheader("Дополнительная оценка manager effect")
    st.caption(
        "Считаем difference-in-differences: изменение отеля после действия минус изменение аналогичных объектов STR/HOTEL. "
        "Региональное разделение не используется."
    )
    st.markdown(
        f"""
        **Окно manager effect теперь берется из типа действия:**  
        Быстрый эффект: месяц ДО = эффективный месяц - 1, месяц ПОСЛЕ = эффективный месяц.  
        Средний эффект: месяц ДО = эффективный месяц - 1, месяц ПОСЛЕ = эффективный месяц + 1.  
        Долгосрочный эффект: месяц ДО = эффективный месяц, месяц ПОСЛЕ = эффективный месяц + 2.  
        **Наложения:** {overlap_policy}.
        """
    )
    with st.expander("Как связаны тип эффекта и manager effect", expanded=False):
        st.markdown(
            """
            1. Subject Action Log автоматически относится к быстрому, среднему или долгосрочному эффекту.
            2. По дате действия определяется эффективный месяц: 1-15 число - текущий месяц, после 15 числа - следующий месяц.
            3. Для типа эффекта выбираются конкретные месяцы ДО и ПОСЛЕ.
            4. Простая таблица Action Logs считает изменение выбранной метрики между этими месяцами.
            5. Manager effect использует те же месяцы, но дополнительно вычитает изменение похожих объектов STR/HOTEL.
            6. Положительный manager effect показывает рост относительно фона, но не доказывает причинность действия сам по себе.
            """
        )
    if not action_impact.empty:
        total_impact_rows = len(action_impact)
        calculable_impact_rows = int(action_impact["manager_effect_pp"].notna().sum())
        overlap_rows = int((action_impact["has_overlap"] == True).sum()) if "has_overlap" in action_impact.columns else 0
        excluded_rows = int((action_impact["included_in_effect"] == False).sum()) if "included_in_effect" in action_impact.columns else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Action Logs в окне", f"{total_impact_rows:,.0f}".replace(",", " "))
        c2.metric("Эффект рассчитан", f"{calculable_impact_rows:,.0f}".replace(",", " "))
        c3.metric("Наложения", f"{overlap_rows:,.0f}".replace(",", " "))
        c4.metric("Исключено", f"{excluded_rows:,.0f}".replace(",", " "))
    if action_impact.empty:
        st.warning("Нет Action Logs после фильтров или не удалось сопоставить их с объектами.")
    else:
        clean = action_impact.dropna(subset=["manager_effect_pp"]).copy()
        if clean.empty:
            st.warning("Не хватает данных до/после действий для расчёта эффекта.")
            excluded_table = action_impact.copy()
            if not excluded_table.empty:
                st.write("Почему строки не вошли в расчёт:")
                excluded_table = excluded_table[[
                    "hotel_id", "action_date", "subject", "outcome", "effect_type_label",
                    "effective_month", "before_month", "after_month",
                    "pre_months_available", "post_months_available",
                    "has_overlap", "overlapping_actions_count",
                    "exclusion_reason", "evaluation_note"
                ]]
                st.dataframe(excluded_table, use_container_width=True, hide_index=True)
        else:
            avg_effect = clean["manager_effect_pp"].mean()
            median_effect = clean["manager_effect_pp"].median()
            positive_share = (clean["manager_effect_pp"] > 0).mean() * 100
            col1, col2, col3 = st.columns(3)
            col1.metric("Средний manager effect, п.п.", f"{avg_effect:.2f}")
            col2.metric("Медианный manager effect, п.п.", f"{median_effect:.2f}")
            col3.metric("Доля положительных", f"{positive_share:.1f}%")
            st.subheader("Manager effect по типу действия")
            by_effect_type = (
                clean.groupby(["effect_type", "effect_type_label"], as_index=False)
                .agg(
                    avg_manager_effect_pp=("manager_effect_pp", "mean"),
                    median_manager_effect_pp=("manager_effect_pp", "median"),
                    positive_share=("manager_effect_pp", lambda values: (values > 0).mean() * 100),
                    actions_count=("manager_effect_pp", "count"),
                )
            )
            effect_order = {"fast": 0, "medium": 1, "long": 2, "unknown": 3}
            by_effect_type["order"] = by_effect_type["effect_type"].map(effect_order).fillna(99)
            by_effect_type = by_effect_type.sort_values("order").drop(columns=["order"])
            st.dataframe(by_effect_type.round(2), use_container_width=True, hide_index=True)
            st.subheader("Эффект по outcome")
            outcome_fig, by_outcome = create_action_outcome_fig(action_impact)
            safe_plotly_chart(outcome_fig, use_container_width=True, key="plot_action_outcome_effect")
            st.dataframe(by_outcome.round(2), use_container_width=True, hide_index=True)
            st.subheader("Эффект по subject")
            by_subject = (
                clean.groupby("subject", as_index=False)
                .agg(
                    avg_manager_effect_pp=("manager_effect_pp", "mean"),
                    median_manager_effect_pp=("manager_effect_pp", "median"),
                    actions_count=("manager_effect_pp", "count")
                )
            )
            by_subject = by_subject[by_subject["actions_count"] >= 3]
            by_subject = by_subject.sort_values("avg_manager_effect_pp", ascending=False)
            st.dataframe(by_subject.round(2), use_container_width=True, hide_index=True)
            st.subheader("Все Action Logs с рассчитанным эффектом")
            action_table = clean.copy()
            for col in ["pre_efficiency", "post_efficiency", "pre_peer_efficiency", "post_peer_efficiency"]:
                action_table[col] = action_table[col].round(3)
            for col in ["hotel_change_pp", "peer_change_pp", "manager_effect_pp"]:
                action_table[col] = action_table[col].round(2)
            action_table = action_table[[
                "hotel_id", "segment_key", "action_date", "subject", "outcome", "effect_type_label",
                "effective_month", "before_month", "after_month",
                "pre_actual", "pre_expected", "post_actual", "post_expected",
                "pre_efficiency", "post_efficiency", "hotel_change_pp", "peer_change_pp",
                "manager_effect_pp", "pre_months_available", "post_months_available",
                "has_overlap", "overlapping_actions_count", "evaluation_note", "verdict"
            ]]
            st.dataframe(action_table, use_container_width=True, hide_index=True)
            st.download_button(
                "Скачать Action Logs impact CSV",
                data=to_csv_bytes(action_table),
                file_name="action_logs_impact.csv",
                mime="text/csv"
            )
            excluded_table = action_impact[action_impact["manager_effect_pp"].isna()].copy()
            if not excluded_table.empty:
                with st.expander("Action Logs, которые не вошли в расчёт эффекта", expanded=False):
                    excluded_table = excluded_table[[
                        "hotel_id", "action_date", "subject", "outcome",
                        "pre_months_available", "post_months_available",
                        "has_overlap", "overlapping_actions_count",
                        "exclusion_reason", "evaluation_note"
                    ]]
                    st.dataframe(excluded_table, use_container_width=True, hide_index=True)
    with st.expander("Справочник subject из ТЗ"):
        subject_ref = pd.DataFrame(
            [{"subject": key, "description": value} for key, value in SUBJECT_DESCRIPTIONS.items()]
        )
        st.dataframe(subject_ref, use_container_width=True, hide_index=True)

with tab7:
    st.subheader("T-статистика manager actions по GBB")
    st.caption(
        "Здесь считается paired t-test: GBB до действия сравнивается с GBB после действия по тому же hotel_id. "
        "T-test показывает статистическую связь, а не гарантированную причинно-следственную зависимость."
    )
    corporate_card(
        "Как читать результат",
        """
        <p><b>p-value &lt; 0.05</b> — изменение считается статистически значимым на уровне 5%.</p>
        <p><b>delta</b> — насколько изменился средний GBB после действия.</p>
        <p><b>n</b> — количество валидных пар hotel_id × action subject после дедупликации.</p>
        <p>Положительный значимый результат можно читать как: на текущих данных действие связано со статистически значимым ростом GBB.</p>
        """,
        badge_text="Paired t-test",
        badge_class="badge"
    )
    if ttest_table.empty:
        st.info("Недостаточно данных для paired t-test после текущих фильтров.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Action Logs", f"{ttest_run_summary['actions_count']:,}".replace(",", " "))
        c2.metric("Пар до дедупликации", f"{ttest_run_summary['valid_pairs_before_dedup']:,}".replace(",", " "))
        c3.metric("Пар в t-test", f"{ttest_run_summary['valid_pairs_count']:,}".replace(",", " "))
        c4.metric("Значимых subject", f"{ttest_run_summary['subjects_significant']:,}".replace(",", " "))
        st.caption(
            f"Исключено пар: {ttest_run_summary['excluded_pairs_count']:,}; "
            f"дублей снято: {ttest_run_summary['duplicate_pairs_removed']:,}; "
            f"subject tested: {ttest_run_summary['subjects_tested']:,}."
        )

        significant = ttest_table[ttest_table["is_significant"] == True].copy()
        if significant.empty:
            st.warning("После текущих фильтров нет действий с p-value < 0.05.")
        else:
            st.markdown("**Главные статистически значимые выводы**")
            for _, row in significant.iterrows():
                if row["platform_status"] == "significant_positive":
                    st.success(
                        f"{row['subject']}: средний GBB вырос с {row['mean_before']:.2f} до {row['mean_after']:.2f} "
                        f"({row['delta_abs']:+.2f}, {row['delta_pct']:+.1f}%), p-value={row['p_value_two_tail']:.5f}, n={int(row['n'])}. "
                        "На текущих данных действие связано со статистически значимым ростом GBB."
                    )
                elif row["platform_status"] == "significant_negative":
                    st.warning(
                        f"{row['subject']}: средний GBB снизился с {row['mean_before']:.2f} до {row['mean_after']:.2f} "
                        f"({row['delta_abs']:+.2f}, {row['delta_pct']:+.1f}%), p-value={row['p_value_two_tail']:.5f}, n={int(row['n'])}. "
                        "Это статистически значимое снижение; действие требует дополнительной проверки."
                    )

        only_significant = st.checkbox("Показывать только статистически значимые действия", value=False)
        shown_t = significant.copy() if only_significant else ttest_table.copy()
        for c in ["mean_before", "mean_after", "delta_abs", "delta_pct", "t_statistic", "df", "p_value_two_tail"]:
            if c in shown_t.columns:
                shown_t[c] = shown_t[c].round(4)
        shown_t["p-value formatted"] = shown_t["p_value_two_tail"].apply(format_pvalue)
        result_cols = [
            "subject", "effect_window", "n", "mean_before", "mean_after",
            "delta_abs", "delta_pct", "t_statistic", "p_value_two_tail",
            "p-value formatted", "is_significant", "direction",
            "platform_status", "platform_summary", "period_min", "period_max",
        ]
        st.dataframe(shown_t[[col for col in result_cols if col in shown_t.columns]], use_container_width=True, hide_index=True)
        st.download_button(
            "Скачать paired t-test CSV",
            data=to_csv_bytes(shown_t),
            file_name="manager_action_gbb_paired_ttest.csv",
            mime="text/csv"
        )

        with st.expander("Техническая диагностика paired t-test", expanded=False):
            st.markdown("**Причины исключения пар**")
            if ttest_diagnostics.empty:
                st.write("Нет исключённых пар.")
            else:
                st.dataframe(ttest_diagnostics, use_container_width=True, hide_index=True)
            st.markdown("**Пары GBB до/после**")
            pairs_shown = ttest_pairs.copy()
            for col in ["action_date"]:
                if col in pairs_shown.columns:
                    pairs_shown[col] = pd.to_datetime(pairs_shown[col], errors="coerce").dt.strftime("%Y-%m-%d")
            pair_cols = [
                "hotel_id", "subject", "outcome", "action_date", "effect_window",
                "effective_month", "month_before", "month_after",
                "gbb_before", "gbb_after", "pair_status", "exclusion_reason",
            ]
            st.dataframe(pairs_shown[[col for col in pair_cols if col in pairs_shown.columns]], use_container_width=True, hide_index=True)

        st.markdown(
            """
            **Ограничение интерпретации:** t-test не доказывает, что действие гарантированно увеличивает бронирования.
            Правильная формулировка: на текущих данных после этого действия наблюдается статистически значимое изменение GBB.
            """
        )

with tab8:
    st.subheader("Кластеризация отелей")
    st.caption(
        "Кластеры строятся только на признаках отеля: масштаб, цена, сезонность, стабильность и динамика. "
        "BizDev-действия присоединяются только после присвоения cluster_id."
    )
    col_a, col_c = st.columns([1.4, 0.8])
    with col_a:
        cluster_mode = st.selectbox(
            "Тип кластеризации",
            ["Комплексная", "Сезонная", "Экономическая"],
            index=0,
            help="Выбирает набор признаков для кластеризации: комплексный бизнес-профиль, сезонность или экономика объекта."
        )
    with col_c:
        cluster_k = st.selectbox(
            "Количество кластеров k",
            [3, 4, 5, 6, 7, 8],
            index=2,
            help="Количество групп, на которые K-means разделит отели. Финальный выбор стоит сверять с профилем кластеров и их размерами."
        )
    with st.spinner("Готовим признаки и запускаем кластеризацию..."):
        with timed_step("clustering.get_features_cached"):
            hotel_features = build_clustering_features_cached(filtered_monthly)
    features_for_cluster = hotel_features.copy()

    if features_for_cluster.empty or features_for_cluster["hotel_id"].nunique() < 3:
        st.warning("Недостаточно отелей для кластеризации после фильтров Блока 2.")
    else:
        clustered_df, features_scaled, cluster_metrics = run_clustering_cached(features_for_cluster, cluster_mode, cluster_k)
        with timed_step("clustering.build_cluster_profile"):
            cluster_profile = build_cluster_profile(clustered_df, mode_name={"Сезонная": "seasonal", "Экономическая": "economic"}.get(cluster_mode, "complex"))
        with timed_step("clustering.attach_bizdev_actions"):
            bizdev_actions_by_cluster = attach_bizdev_actions(clustered_df, filtered_actions)
        with timed_step("clustering.bizdev_effect_by_cluster"):
            bizdev_effect_by_cluster = build_bizdev_effect_by_cluster(clustered_df, action_impact)

        min_cluster_size = int(cluster_profile["hotels_count"].min()) if not cluster_profile.empty else 0
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Отелей в кластеризации", format_number(clustered_df["hotel_id"].nunique()))
        m2.metric("Фактических кластеров", format_number(clustered_df["cluster_id"].nunique()))
        m3.metric("Silhouette score", format_number(cluster_metrics.get("silhouette_score"), 3))
        m4.metric("Мин. размер кластера", format_number(min_cluster_size))
        st.caption(
            f"k = {cluster_k}. База кластеризации: выборка Блока 2 за {window_start.strftime('%Y-%m')} → {window_end.strftime('%Y-%m')}. "
            f"В кластеризацию передано {clustered_df['hotel_id'].nunique():,} отелей из этой выборки."
        )
        if min_cluster_size < 20:
            st.warning(
                f"Есть маленький кластер: {min_cluster_size} отелей. "
                "Для бизнес-интерпретации такой кластер может быть слишком шумным. Попробуй уменьшить k."
            )
        render_cluster_takeaways(
            "Короткий вывод по кластеризации",
            build_cluster_quality_takeaways(cluster_profile, cluster_metrics, cluster_k)
        )
        render_method_note(
            "Как читать качество кластеризации",
            "При изменении k кластеризация пересчитывается заново: меняются центры K-means, cluster_id, названия кластеров, таблицы и все графики. Silhouette score показывает, насколько отели внутри одного кластера похожи друг на друга и отличаются от соседних кластеров. Очень маленькие кластеры стоит трактовать осторожно или проверять на меньшем k."
        )

        st.markdown("### Таблица признаков")
        feature_cols = [
            "hotel_id", "cluster_id", "cluster_name", "cluster_explanation", "is_STR",
            "avg_monthly_revenue", "avg_monthly_roomnights", "avg_monthly_gbb",
            "ADR", "LOS", "SI", "CV", "summer_share", "winter_share",
            "growth_2025_vs_2024", "months_available"
        ]
        feature_table = clustered_df[[c for c in feature_cols if c in clustered_df.columns]].copy()
        numeric_cols = feature_table.select_dtypes(include=[np.number]).columns
        feature_table[numeric_cols] = feature_table[numeric_cols].round(3)
        st.dataframe(feature_table, use_container_width=True, hide_index=True)
        render_cluster_explanations("Что означают столбцы таблицы признаков", feature_table.columns)

        st.markdown("### Профиль кластеров")
        profile_shown = cluster_profile.copy()
        numeric_cols = profile_shown.select_dtypes(include=[np.number]).columns
        profile_shown[numeric_cols] = profile_shown[numeric_cols].round(3)
        st.dataframe(profile_shown, use_container_width=True, hide_index=True)
        render_cluster_takeaways("Что получилось по кластерам", build_cluster_profile_takeaways(cluster_profile))
        render_cluster_explanations('Что означают столбцы профиля кластеров', profile_shown.columns)

        st.markdown("### Сезонность по кластерам")
        cluster_lookup = clustered_df[["hotel_id", "cluster_id", "cluster_name"]].drop_duplicates("hotel_id")
        cluster_monthly = filtered_monthly.merge(cluster_lookup, on="hotel_id", how="inner")
        seasonality_cluster_df = calculate_seasonality_improved(
            cluster_monthly,
            "roomnights",
            group_cols=["cluster_id", "cluster_name"],
        )
        seasonality_cluster_df["month_name"] = seasonality_cluster_df["month_num"].map(MONTH_NAMES)
        seasonality_fig = px.line(
            seasonality_cluster_df,
            x="month_name", y="seasonality_index", color="cluster_name",
            markers=True,
            labels={
                "month_name": "Месяц",
                "seasonality_index": "Сезонный индекс",
                "cluster_name": "Кластер",
                "hotel_count": "Отелей в месяце",
            },
            hover_data=["hotel_count"],
            title="Сезонный индекс roomnights по кластерам"
        )
        seasonality_fig.update_layout(xaxis={"categoryorder": "array", "categoryarray": list(MONTH_NAMES.values())})
        safe_plotly_chart(seasonality_fig, use_container_width=True, key="plot_hotel_cluster_seasonality")
        render_cluster_takeaways("Что видно по сезонности", build_seasonality_takeaways(seasonality_cluster_df))
        render_method_note("Вывод по сезонности", "График считается по тому же принципу, что и вкладка «Сезонность»: для каждого месяца берётся среднее по отелям, месяц взвешивается по количеству отелей, а аномальные месяцы исключаются по Z-score. Если индекс выше 1, кластер сильнее своего среднего уровня; если ниже 1, месяц слабее среднего.")
        st.markdown("### PCA scatter plot")
        pca_fig = px.scatter(
            clustered_df,
            x="pca_1", y="pca_2", color="cluster_name",
            hover_data=["hotel_id", "cluster_id", "is_STR", "avg_monthly_revenue", "ADR", "LOS", "months_available"],
            labels={"pca_1": "PCA component 1", "pca_2": "PCA component 2", "cluster_name": "Кластер"},
            title="PCA-карта отелей"
        )
        pca_fig.update_layout(height=560)
        safe_plotly_chart(pca_fig, use_container_width=True, key="plot_hotel_cluster_pca")
        render_method_note("Вывод по PCA", "PCA сжимает множество признаков до двух осей, чтобы визуально проверить разделение кластеров. Хороший признак - группы заметно отделяются друг от друга. Если точки сильно перемешаны, кластеры могут быть близкими по бизнес-профилю, и k стоит перепроверить по таблице профиля.")
        st.markdown("### Heatmap признаков по кластерам")
        heatmap_features = [
            "avg_monthly_revenue", "avg_monthly_roomnights", "avg_monthly_gbb",
            "ADR", "LOS", "SI", "CV", "summer_share", "winter_share", "growth_2025_vs_2024"
        ]
        heat = cluster_profile.set_index("cluster_name")[[c for c in heatmap_features if c in cluster_profile.columns]].copy()
        heat_z = (heat - heat.mean()) / heat.std(ddof=0).replace(0, 1)
        heat_fig = px.imshow(
            heat_z.fillna(0),
            text_auto=".1f",
            aspect="auto",
            color_continuous_scale=["#59758C", "#FFFFFF", "#CC7D5E"],
            labels={"color": "Z-score"},
            title="Отклонение признаков кластера от среднего"
        )
        safe_plotly_chart(heat_fig, use_container_width=True, key="plot_hotel_cluster_heatmap")
        render_cluster_takeaways("Отличительные признаки кластеров", build_heatmap_takeaways(heat_z))
        render_method_note("Вывод по heatmap", "Heatmap показывает отклонения признаков кластера от среднего по всей выборке. Рыжий цвет означает, что кластер выше среднего по признаку, синий - ниже среднего. Это главный график для объяснения, чем один кластер бизнесово отличается от другого.")
        st.markdown("### Размеры кластеров")
        size_fig = px.bar(
            cluster_profile,
            x="cluster_name", y="hotels_count", text="hotels_count",
            labels={"cluster_name": "Кластер", "hotels_count": "Количество отелей"},
            title="Количество отелей в каждом кластере"
        )
        safe_plotly_chart(size_fig, use_container_width=True, key="plot_hotel_cluster_sizes")
        render_method_note("Вывод по размерам кластеров", "График нужен для проверки устойчивости типологии. Очень маленький кластер может быть реальной нишевой группой, но также может быть выбросом. Если один кластер забирает почти всю выборку, выбранное k может быть слишком большим или признаки плохо разделяют объекты.")
        st.markdown("### BizDev effect by cluster")
        if bizdev_effect_by_cluster.empty:
            st.info("Нет рассчитанных BizDev-действий для присоединения к кластерам.")
        else:
            bizdev_shown = bizdev_effect_by_cluster.copy()
            numeric_cols = bizdev_shown.select_dtypes(include=[np.number]).columns
            bizdev_shown[numeric_cols] = bizdev_shown[numeric_cols].round(3)
            st.dataframe(bizdev_shown, use_container_width=True, hide_index=True)
            render_cluster_explanations("Что означают столбцы BizDev effect by cluster", bizdev_shown.columns)
            render_method_note("Вывод по BizDev внутри кластеров", "Эта таблица не влияет на построение кластеров. Сначала отели получают cluster_id по собственным признакам, и только потом Action Logs присоединяются по hotel_id. Сравнивайте subject, средний/медианный эффект и success rate внутри каждого кластера: так видно, какие действия лучше работают для разных типов объектов.")
        export_cols = [
            "hotel_id", "cluster_id", "cluster_name", "cluster_explanation", "is_STR",
            "avg_monthly_revenue", "avg_monthly_sales", "avg_monthly_roomnights", "avg_monthly_gbb",
            "total_revenue_24m", "ADR", "sales_per_roomnight", "revenue_per_booking", "LOS",
            "SI", "CV", "top3_share", "summer_share", "winter_share", "peak_month",
            "growth_revenue_yoy", "growth_roomnights_yoy", "growth_gbb_yoy",
            "median_monthly_yoy", "months_available", "active_months", "zero_months", "has_full_24m"
        ]
        cluster_export = clustered_df[[c for c in export_cols if c in clustered_df.columns]].copy()
        st.download_button(
            "Скачать результат кластеризации CSV",
            data=to_csv_bytes(cluster_export),
            file_name="hotel_clustering_result.csv",
            mime="text/csv"
        )

with tab10:
    st.subheader("Методика анализа и глоссарий")
    st.markdown(
        """
        <div class="corporate-card">
            <b>Быстрый переход к формулам:</b><br>
            <span id="method-four-metrics"></span><a href="#method-four-metrics">4 метрики</a> ·
            <span id="method-seasonality"></span><a href="#method-seasonality">сезонность</a> ·
            <span id="method-expected"></span><a href="#method-expected">expected</a> ·
            <span id="method-efficiency"></span><a href="#method-efficiency">efficiency</a> ·
            <span id="method-individual-effect"></span><a href="#method-individual-effect">individual effect</a> ·
            <span id="method-action-window"></span><a href="#method-action-window">окно Action Log</a> ·
            <span id="method-overlap"></span><a href="#method-overlap">наложения</a> ·
            <span id="method-manager-effect"></span><a href="#method-manager-effect">manager effect</a> ·
            <span id="method-correlation"></span><a href="#method-correlation">корреляции</a> ·
            <span id="method-ttest"></span><a href="#method-ttest">t-статистика</a> ·
            <span id="method-clustering"></span><a href="#method-clustering">кластеризация</a>
            <hr>
            <p><b>seasonality_index</b> = среднее значение календарного месяца / среднее значение всех месяцев</p>
            <p><b>expected</b> = hotel_baseline × seasonality_index</p>
            <p><b>hotel_efficiency</b> = actual / expected</p>
            <p><b>individual_effect</b> = (hotel_efficiency - 1) - (segment_efficiency - 1)</p>
            <p><b>manager_effect</b> = (post_hotel_eff - pre_hotel_eff) - (post_peer_eff - pre_peer_eff)</p>
            <p><b>t-statistic</b> = разница средних / стандартная ошибка разницы</p>
            <p><b>weighted clustering</b> = стандартизированные признаки × выбранные веса → k-means</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    with st.expander("Глоссарий английских терминов", expanded=False):
        show_glossary()
    st.markdown(
        f"""
        ### Короткая логика проекта
        Цель алгоритма – на уровне конкретного отеля отделить естественный рост одной локалии от индивидуального роста, который может быть связан с работой менеджера.
        В проекте не используется региональное разделение. Вся база считается одной локалией.
        ### Step-by-step guide для документации
        **Шаг 1. Подготовить данные продаж**
        Исходная таблица продаж содержит строки с hotel_id, датой бронирования и метриками.
        **Шаг 2. Агрегировать продажи до уровня hotel_id × месяц**
        **Шаг 3. Выбрать главную метрику анализа** – {METRICS[metric]}.
        **Шаг 4. Рассчитать сезонность одной локалии**
        seasonality_index = weighted_average_calendar_month_value / weighted_average_all_months_value, где вес месяца = количество отелей; месяцы-выбросы по Z-score > 3 исключаются.
        **Шаг 5. Посчитать ожидаемое значение для каждого отеля**
        expected = hotel_baseline × seasonality_index
        **Шаг 6. Посчитать эффективность отеля** hotel_efficiency = actual / expected
        **Шаг 7. Посчитать фон похожих объектов**
        **Шаг 8. Посчитать индивидуальный эффект объекта**
        individual_effect = hotel_residual - segment_residual
        **Шаг 9. Настроить окно оценки Action Log**
        Текущее окно: {pre_window_months} мес. до → месяц действия исключается → лаг {lag_months} мес. → {post_window_months} мес. после
        **Шаг 10. Проверить наложение действий**
        **Шаг 11. Рассчитать manager effect**
        manager_effect = hotel_change - peer_change
        **Шаг 12. Интерпретировать результат**
        **Шаг 13. Проверить статистическую значимость через t-статистику**
        **Шаг 14. Построить кластеризацию по всем метрикам**
        ### Как ползунки влияют на генеральную совокупность
        **Тип объекта** меняет саму базу отелей.
        **Outcome и Subject** не удаляют продажи, но меняют набор Action Logs.
        **Период до / лаг / период после** меняют количество Action Logs, по которым можно рассчитать эффект.
        ### Ограничение метода
        Модель не доказывает причинность на 100%. Это аналитический прокси.
        """
    )
