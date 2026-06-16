import pandas as pd
import streamlit as st

from action_log_effect_config import EFFECT_TYPE_LABELS
from utils import to_csv_bytes


EXCLUSION_REASON_LABELS = {
    "unknown_subject": "Subject отсутствует в справочнике типов эффекта",
    "invalid_action_date": "Некорректная дата Action Log",
    "missing_hotel_id": "Отсутствует hotel_id",
    "missing_subject": "Отсутствует subject",
    "missing_before_month": "Нет месяца ДО",
    "missing_after_month": "Нет месяца ПОСЛЕ",
    "missing_before_and_after_month": "Нет месяцев ДО и ПОСЛЕ",
    "missing_metric_before": "Нет значения метрики в месяце ДО",
    "missing_metric_after": "Нет значения метрики в месяце ПОСЛЕ",
}


def _count_reason(effect_df, reason):
    if effect_df.empty or "exclusion_reason" not in effect_df.columns:
        return 0
    return int(effect_df["exclusion_reason"].fillna("").str.contains(reason, regex=False).sum())


def render_action_log_effect_ui(effect_df, metric_label):
    st.subheader("Автоматическая оценка изменения после Action Logs")
    st.info(
        "Тип эффекта автоматически определяется по типу действия. Эффективный месяц: "
        "1–15 число — текущий месяц; после 15 числа — следующий месяц."
    )
    st.caption(
        "Корректная интерпретация: изменение выбранной метрики между месяцем ДО и месяцем ПОСЛЕ, "
        "определёнными по методологии Action Logs. Положительное изменение не доказывает причинный эффект."
    )

    if effect_df is None or effect_df.empty:
        st.warning("Нет Action Logs после фильтров для автоматического расчёта.")
        return

    total_rows = len(effect_df)
    calculated_rows = int((effect_df["calculation_status"] == "calculated").sum())
    excluded_rows = total_rows - calculated_rows
    hotels_in_calc = effect_df.loc[effect_df["calculation_status"] == "calculated", "hotel_id"].nunique()
    known_type_rows = int(effect_df["effect_type"].ne("unknown").sum())
    unknown_type_rows = int(effect_df["effect_type"].eq("unknown").sum())
    found_before_rows = int(effect_df["metric_before"].notna().sum())
    found_after_rows = int(effect_df["metric_after"].notna().sum())
    found_both_rows = int(effect_df["metric_before"].notna().mul(effect_df["metric_after"].notna()).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Action Logs после фильтров", f"{total_rows:,}".replace(",", " "))
    c2.metric("Action Logs рассчитано", f"{calculated_rows:,}".replace(",", " "))
    c3.metric("Action Logs исключено", f"{excluded_rows:,}".replace(",", " "))
    c4.metric("Отелей в расчёте", f"{hotels_in_calc:,}".replace(",", " "))

    distribution = (
        effect_df.groupby(["effect_type", "effect_type_label"], as_index=False)
        .size()
        .rename(columns={"size": "Action Logs"})
    )
    effect_order = ["fast", "medium", "long", "unknown"]
    distribution["order"] = distribution["effect_type"].apply(lambda value: effect_order.index(value) if value in effect_order else 99)
    distribution = distribution.sort_values("order").drop(columns=["order"])
    st.markdown("**Распределение по типам эффекта**")
    st.dataframe(distribution, use_container_width=True, hide_index=True)

    unknown_subjects = (
        effect_df.loc[effect_df["effect_type"].eq("unknown")]
        .groupby(["subject", "subject_normalized"], dropna=False)
        .size()
        .reset_index(name="action_logs_count")
        .sort_values("action_logs_count", ascending=False)
    )
    if not unknown_subjects.empty:
        st.warning(f"Не определён тип эффекта для {unknown_type_rows} Action Logs.")

    with st.expander("Методология расчёта", expanded=False):
        methodology = pd.DataFrame([
            {"Тип эффекта": EFFECT_TYPE_LABELS["fast"], "Месяц ДО": "эффективный месяц − 1", "Месяц ПОСЛЕ": "эффективный месяц"},
            {"Тип эффекта": EFFECT_TYPE_LABELS["medium"], "Месяц ДО": "эффективный месяц − 1", "Месяц ПОСЛЕ": "эффективный месяц + 1"},
            {"Тип эффекта": EFFECT_TYPE_LABELS["long"], "Месяц ДО": "эффективный месяц", "Месяц ПОСЛЕ": "эффективный месяц + 2"},
        ])
        st.dataframe(methodology, use_container_width=True, hide_index=True)
        st.markdown(
            """
            Тип эффекта задаёт только окно сравнения. Он не означает, что эффект точно существует,
            что изменение вызвано именно Action Log, что изменение статистически значимо или что исключено влияние сезонности.
            """
        )

    with st.expander("Диагностика расчёта", expanded=False):
        st.markdown("**Контрольные значения pipeline**")
        control_values = pd.DataFrame([
            {"Показатель": "Action Logs после фильтров", "Количество": total_rows},
            {"Показатель": "Тип эффекта определён", "Количество": known_type_rows},
            {"Показатель": "Тип эффекта не определён", "Количество": unknown_type_rows},
            {"Показатель": "Найдено значение ДО", "Количество": found_before_rows},
            {"Показатель": "Найдено значение ПОСЛЕ", "Количество": found_after_rows},
            {"Показатель": "Найдены оба значения", "Количество": found_both_rows},
            {"Показатель": "Рассчитано", "Количество": calculated_rows},
            {"Показатель": "Исключено", "Количество": excluded_rows},
            {"Показатель": "Проверка: рассчитано + исключено", "Количество": calculated_rows + excluded_rows},
        ])
        st.dataframe(control_values, use_container_width=True, hide_index=True)

        st.markdown("**Причины исключения**")
        diagnostics = (
            effect_df.loc[effect_df["calculation_status"].eq("excluded"), "exclusion_reason"]
            .fillna("unknown")
            .value_counts()
            .rename_axis("Причина исключения")
            .reset_index(name="Количество Action Logs")
        )
        if not diagnostics.empty:
            diagnostics["Описание"] = diagnostics["Причина исключения"].map(EXCLUSION_REASON_LABELS).fillna(diagnostics["Причина исключения"])
        else:
            diagnostics = pd.DataFrame(columns=["Причина исключения", "Количество Action Logs", "Описание"])
        zero_before_count = int(effect_df["zero_before_value"].fillna(False).sum())
        st.dataframe(diagnostics, use_container_width=True, hide_index=True)
        st.write(f"Нулевая база metric_before: {zero_before_count:,}".replace(",", " "))

        if not unknown_subjects.empty:
            st.markdown("**Неизвестные subject после нормализации**")
            c_unknown_1, c_unknown_2 = st.columns(2)
            c_unknown_1.metric("Неизвестных subject", f"{unknown_subjects['subject_normalized'].nunique():,}".replace(",", " "))
            c_unknown_2.metric("Action Logs с неизвестным subject", f"{unknown_type_rows:,}".replace(",", " "))
            st.dataframe(
                unknown_subjects.rename(columns={
                    "subject": "Исходный subject",
                    "subject_normalized": "Нормализованный subject",
                    "action_logs_count": "Количество Action Logs",
                }),
                use_container_width=True,
                hide_index=True,
            )

        excluded = effect_df[effect_df["calculation_status"] == "excluded"].copy()
        if not excluded.empty:
            st.write("Исключённые Action Logs с причинами:")
            excluded_cols = [
                "hotel_id", "action_date", "subject", "outcome",
                "effect_type_label", "before_month", "after_month", "exclusion_reason"
            ]
            st.dataframe(excluded[excluded_cols], use_container_width=True, hide_index=True)

    shown = effect_df.copy()
    date_cols = ["action_date", "effective_month", "before_month", "after_month"]
    for col in date_cols:
        if col in shown.columns:
            shown[col] = pd.to_datetime(shown[col], errors="coerce").dt.strftime("%Y-%m-%d")
    numeric_cols = ["metric_before", "metric_after", "absolute_change", "percentage_change"]
    for col in numeric_cols:
        if col in shown.columns:
            shown[col] = shown[col].round(2)
    st.markdown(f"**Итоговая таблица расчёта по метрике: {metric_label}**")
    table_cols = [
        "hotel_id", "action_date", "subject", "outcome",
        "effect_type_label", "effective_month", "before_month", "after_month",
        "metric_name", "metric_before", "metric_after",
        "absolute_change", "percentage_change", "change_direction_label",
        "zero_before_value", "calculation_status", "exclusion_reason",
    ]
    st.dataframe(shown[[col for col in table_cols if col in shown.columns]], use_container_width=True, hide_index=True)
    st.download_button(
        "Скачать автоматический расчёт Action Logs CSV",
        data=to_csv_bytes(effect_df),
        file_name="automatic_action_log_effects.csv",
        mime="text/csv",
    )
