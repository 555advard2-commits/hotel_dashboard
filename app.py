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
    calculate_seasonality_by_segment
)
from analysis import (
    evaluate_sample_quality, calculate_period_classification,
    calculate_action_impact
)
from visualization import (
    create_monthly_actual_expected_fig, create_all_metrics_seasonality_fig,
    create_multi_metric_index_fig,
    create_action_outcome_fig, create_ttest_bar_fig
)
from statistics import (
    build_ttest_tables, format_pvalue, pvalue_interpretation
)
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


inject_global_css()
apply_corporate_plotly_theme()

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
    sample_quality, prepared_monthly_strict = evaluate_sample_quality(
        monthly_df, selected_metrics=sample_metrics,
        window_start=window_start, window_end=window_end,
        require_full_observed_span=require_full_observed_span,
        min_nonzero_months=min_nonzero_months,
        max_zero_months=max_zero_months,
        max_zero_share_pct=max_zero_share_pct,
        min_observed_months=min_observed_months,
        strict_all_selected_metrics=True
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
with timed_step("model.add_expected_values"):
    enriched_df = add_expected_values(filtered_monthly, metric)
with timed_step("model.calculate_hotel_scores"):
    hotel_scores = calculate_hotel_scores(enriched_df, metric)
with timed_step("model.calculate_metric_summary"):
    metric_summary = calculate_metric_summary(filtered_monthly)
with timed_step("analysis.calculate_period_classification"):
    period_classification = calculate_period_classification(
        enriched_df, filtered_actions, metric, graining, threshold
    )
with timed_step("analysis.calculate_action_impact"):
    action_impact = calculate_action_impact(
        filtered_actions, enriched_df, metric,
        pre_window_months, post_window_months, lag_months,
        min_months_per_side, overlap_policy
    )
with timed_step("statistics.build_ttest_tables"):
    ttest_table = build_ttest_tables(period_classification, action_impact)
with timed_step("visualization.create_ttest_bar_fig"):
    ttest_fig = create_ttest_bar_fig(ttest_table)

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
tab1, tab2, tab4, tab7, tab8, tab10 = st.tabs([
    "Обзор 4 метрик",
    "Сезонность",
    "Action Logs",
    "T-статистика",
    "Кластеризация",
    "Методика и глоссарий"
])

with tab1:
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

with tab2:
    st.subheader("Сезонность по всем основным метрикам")
    fig, seasonality_table = create_all_metrics_seasonality_fig(filtered_monthly)
    safe_plotly_chart(fig, use_container_width=True, key="plot_seasonality_all_metrics")
    st.caption("Индекс 1.20 = месяц на 20% сильнее среднего. Индекс 0.80 = месяц на 20% слабее среднего.")
    seasonality_table["seasonality_index"] = seasonality_table["seasonality_index"].round(3)
    st.dataframe(
        seasonality_table[["metric", "month_num", "month_name", "metric_value", "seasonality_index"]],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Сезонность по типу объекта")
    st.caption("Регионов нет. Сравнение делается только внутри одной локалии по типу объекта: STR / HOTEL.")
    seg_seasonality = calculate_seasonality_by_segment(filtered_monthly, metric)
    seg_seasonality["month_name"] = seg_seasonality["month_num"].map(MONTH_NAMES)
    fig = px.line(
        seg_seasonality,
        x="month_name", y="seasonality_index", color="segment_key",
        markers=True,
        labels={"month_name": "Месяц", "seasonality_index": "Сезонный индекс", "segment_key": "Тип объекта"}
    )
    fig.update_layout(xaxis={"categoryorder": "array", "categoryarray": MONTH_ORDER})
    safe_plotly_chart(fig, use_container_width=True, key="plot_seasonality_by_segment")

with tab4:
    st.subheader("Оценка влияния Action Logs")
    st.caption(
        "Считаем difference-in-differences: изменение отеля после действия минус изменение аналогичных объектов STR/HOTEL. "
        "Региональное разделение не используется."
    )
    st.markdown(
        f"""
        **Текущее окно оценки:** {pre_window_months} мес. до → месяц действия исключён → лаг {lag_months} мес. → {post_window_months} мес. после.  
        **Минимум для расчёта:** не меньше {min_months_per_side} доступных месяцев до и после.  
        **Наложения:** {overlap_policy}.
        """
    )
    with st.expander("Почему окно оценки устроено именно так", expanded=False):
        st.markdown(
            """
            1. Сначала берём период **до действия**, чтобы понять нормальный уровень объекта до менеджерской проработки.
            2. Месяц самого Action Log исключаем, потому что действие могло быть сделано в любой день месяца.
            3. **Лаг** нужен, если действие не может сработать сразу.
            4. Период **после действия** показывает результат. Базовый вариант – 3 месяца.
            5. Если у того же отеля в это окно попали другие Action Logs, эффект считается смешанным.
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
                    "hotel_id", "action_date", "subject", "outcome",
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
                "hotel_id", "segment_key", "action_date", "subject", "outcome",
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
    st.subheader("T-статистика: проверка, эффект или шум")
    st.caption(
        "T-test отвечает не на вопрос 'красиво ли выросло', а на вопрос: отличается ли средний эффект от нуля или от другой группы сильнее, чем можно ожидать от случайного разброса."
    )
    corporate_card(
        "Как читать t-statistic",
        """
        <p><b>t-statistic</b> показывает, насколько наблюдаемая разница велика относительно разброса данных.</p>
        <p><b>p-value</b> показывает, насколько вероятно получить такую разницу случайно, если реального эффекта нет.</p>
        <p>Обычно <b>p-value &lt; 0.05</b> трактуют как статистически заметное отличие.</p>
        """,
        badge_text="Статистическая проверка",
        badge_class="badge"
    )
    if ttest_table.empty:
        st.info("Недостаточно данных для t-статистики после текущих фильтров.")
    else:
        if ttest_fig is not None:
            safe_plotly_chart(ttest_fig, use_container_width=True, key="plot_ttest_bar")
        shown_t = ttest_table.copy()
        for c in ["Среднее A", "Среднее B", "Разница средних", "t-statistic", "df", "p-value"]:
            if c in shown_t.columns:
                shown_t[c] = shown_t[c].round(4)
        shown_t["p-value formatted"] = shown_t["p-value"].apply(format_pvalue)
        st.dataframe(shown_t, use_container_width=True, hide_index=True)
        st.download_button(
            "Скачать t-статистику CSV",
            data=to_csv_bytes(shown_t),
            file_name="ttest_analysis.csv",
            mime="text/csv"
        )
    st.markdown(
        """
        Что здесь проверяется:
        - отличается ли средний manager effect после Action Logs от нуля;
        - отличаются ли успешные outcomes от негативных;
        - отличаются ли действия без наложений от действий с наложениями;
        - отличаются ли объект-периоды с Action Logs от объект-периодов без Action Logs.
        """
    )

with tab8:
    st.subheader("Кластеризация отелей")
    st.caption(
        "Кластеры строятся только на признаках отеля: масштаб, цена, сезонность, стабильность и динамика. "
        "BizDev-действия присоединяются только после присвоения cluster_id."
    )
    col_a, col_b, col_c, col_d = st.columns([1.2, 1.2, 0.8, 0.9])
    with col_a:
        cluster_mode = st.selectbox(
            "Тип кластеризации",
            ["Комплексная", "Сезонная", "Экономическая"],
            index=0,
            help="Выбирает набор признаков для кластеризации: комплексный бизнес-профиль, сезонность или экономика объекта."
        )
    with col_b:
        cluster_sample = st.selectbox(
            "Выборка данных",
            ["Полные 24 месяца", "20+ месяцев", "12+ месяцев", "Все доступные"],
            index=1,
            help="Ограничивает отели по количеству месяцев с данными. Чем строже выборка, тем надёжнее сезонные признаки."
        )
    with col_c:
        cluster_k = st.selectbox(
            "Количество кластеров k",
            [3, 4, 5, 6, 7, 8],
            index=2,
            help="Количество групп, на которые K-means разделит отели. Финальный выбор стоит сверять с профилем кластеров и их размерами."
        )
    with col_d:
        run_cluster = st.button(
            "Запустить кластеризацию",
            use_container_width=True,
            help="Считает признаки отелей, запускает кластеризацию и строит таблицы/графики. Action Logs не входят в признаки."
        )

    if run_cluster:
        with st.spinner("Ð“Ð¾Ñ‚Ð¾Ð²Ð¸Ð¼ Ð¿Ñ€Ð¸Ð·Ð½Ð°ÐºÐ¸ Ð¸ Ð·Ð°Ð¿ÑƒÑÐºÐ°ÐµÐ¼ ÐºÐ»Ð°ÑÑ‚ÐµÑ€Ð¸Ð·Ð°Ñ†Ð¸ÑŽ..."):
            with timed_step("clustering.get_features_cached"):
                hotel_features = build_clustering_features_cached(filtered_monthly)
    else:
        hotel_features = pd.DataFrame(columns=["hotel_id", "months_available", "has_full_24m"])
    min_months_by_sample = {
        "Полные 24 месяца": 24,
        "20+ месяцев": 20,
        "12+ месяцев": 12,
        "Все доступные": 1,
    }
    with timed_step("clustering.apply_sample_filter"):
        min_months = min_months_by_sample[cluster_sample]
        features_for_cluster = hotel_features[hotel_features["months_available"] >= min_months].copy()
        if cluster_sample == "Полные 24 месяца":
            features_for_cluster = features_for_cluster[features_for_cluster["has_full_24m"] == True].copy()

    if not run_cluster:
        st.info("Выбери настройки и нажми «Запустить кластеризацию».")
    elif features_for_cluster.empty or features_for_cluster["hotel_id"].nunique() < 3:
        st.warning("Недостаточно отелей для кластеризации после выбранного ограничения по месяцам.")
    else:
        with timed_step("clustering.run_algorithm"):
            if cluster_mode == "Сезонная":
                clustered_df, features_scaled, cluster_metrics = run_seasonal_clustering(features_for_cluster, cluster_k)
            elif cluster_mode == "Экономическая":
                clustered_df, features_scaled, cluster_metrics = run_economic_clustering(features_for_cluster, cluster_k)
            else:
                clustered_df, features_scaled, cluster_metrics = run_complex_clustering(features_for_cluster, cluster_k)
        with timed_step("clustering.build_cluster_profile"):
            cluster_profile = build_cluster_profile(clustered_df)
        with timed_step("clustering.attach_bizdev_actions"):
            bizdev_actions_by_cluster = attach_bizdev_actions(clustered_df, filtered_actions)
        with timed_step("clustering.bizdev_effect_by_cluster"):
            bizdev_effect_by_cluster = build_bizdev_effect_by_cluster(clustered_df, action_impact)

        m1, m2, m3 = st.columns(3)
        m1.metric("Отелей в кластеризации", format_number(clustered_df["hotel_id"].nunique()))
        m2.metric("Silhouette score", format_number(cluster_metrics.get("silhouette_score"), 3))
        m3.metric("Inertia / Elbow", format_number(cluster_metrics.get("inertia"), 2))

        st.markdown("### Таблица признаков")
        feature_cols = [
            "hotel_id", "cluster_id", "cluster_name", "is_STR",
            "avg_monthly_revenue", "avg_monthly_roomnights", "avg_monthly_gbb",
            "ADR", "LOS", "SI", "CV", "summer_share", "winter_share",
            "growth_2025_vs_2024", "months_available"
        ]
        feature_table = clustered_df[[c for c in feature_cols if c in clustered_df.columns]].copy()
        numeric_cols = feature_table.select_dtypes(include=[np.number]).columns
        feature_table[numeric_cols] = feature_table[numeric_cols].round(3)
        st.dataframe(feature_table, use_container_width=True, hide_index=True)

        st.markdown("### Профиль кластеров")
        profile_shown = cluster_profile.copy()
        numeric_cols = profile_shown.select_dtypes(include=[np.number]).columns
        profile_shown[numeric_cols] = profile_shown[numeric_cols].round(3)
        st.dataframe(profile_shown, use_container_width=True, hide_index=True)

        st.markdown("### Сезонность по кластерам")
        seasonality_rows = []
        for cluster_id, group in clustered_df.groupby("cluster_id"):
            cluster_name = group["cluster_name"].iloc[0]
            for month_num in range(1, 13):
                seasonality_rows.append({
                    "cluster_id": cluster_id,
                    "cluster_name": cluster_name,
                    "month_num": month_num,
                    "month_name": MONTH_NAMES.get(month_num, str(month_num)),
                    "roomnights_share": group[f"roomnights_share_m{month_num:02d}"].mean(),
                    "gbb_share": group[f"gbb_share_m{month_num:02d}"].mean(),
                    "revenue_share": group[f"revenue_share_m{month_num:02d}"].mean(),
                })
        seasonality_cluster_df = pd.DataFrame(seasonality_rows)
        seasonality_fig = px.line(
            seasonality_cluster_df,
            x="month_name", y="roomnights_share", color="cluster_name",
            markers=True,
            labels={"month_name": "Месяц", "roomnights_share": "Средняя доля спроса", "cluster_name": "Кластер"},
            title="Средний сезонный профиль кластеров"
        )
        seasonality_fig.update_layout(xaxis={"categoryorder": "array", "categoryarray": list(MONTH_NAMES.values())})
        safe_plotly_chart(seasonality_fig, use_container_width=True, key="plot_hotel_cluster_seasonality")

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

        st.markdown("### Размеры кластеров")
        size_fig = px.bar(
            cluster_profile,
            x="cluster_name", y="hotels_count", text="hotels_count",
            labels={"cluster_name": "Кластер", "hotels_count": "Количество отелей"},
            title="Количество отелей в каждом кластере"
        )
        safe_plotly_chart(size_fig, use_container_width=True, key="plot_hotel_cluster_sizes")

        st.markdown("### BizDev effect by cluster")
        if bizdev_effect_by_cluster.empty:
            st.info("Нет рассчитанных BizDev-действий для присоединения к кластерам.")
        else:
            bizdev_shown = bizdev_effect_by_cluster.copy()
            numeric_cols = bizdev_shown.select_dtypes(include=[np.number]).columns
            bizdev_shown[numeric_cols] = bizdev_shown[numeric_cols].round(3)
            st.dataframe(bizdev_shown, use_container_width=True, hide_index=True)

        export_cols = [
            "hotel_id", "cluster_id", "cluster_name", "is_STR",
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
        seasonality_index = average_calendar_month_value / average_all_months_value
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
