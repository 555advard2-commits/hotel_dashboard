from datetime import datetime

import html
import numpy as np
import pandas as pd
import plotly.io as pio

from config import METRICS, METHODOLOGY_LINKS
from utils import safe_divide, format_number, format_percent, to_csv_bytes
from visualization import (
    create_monthly_actual_expected_fig,
    create_all_metrics_seasonality_fig,
    create_multi_metric_index_fig,
    create_verdict_fig,
    create_worked_fig,
    create_action_outcome_fig,
    create_correlation_heatmap,
)


def build_main_answer(period_classification, action_impact, worked_summary, threshold_percent):
    total_rows = len(period_classification)
    if total_rows == 0:
        return {
            "verdict": "Недостаточно данных",
            "badge_class": "badge-warn",
            "summary": "После фильтров не осталось строк объект-период.",
            "stats": {}
        }
    manager_rows = period_classification[period_classification["verdict"].str.contains("вероятен вклад менеджера", na=False)]
    seasonal_rows = period_classification[period_classification["verdict"].str.contains("сезонный|локальный фон", case=False, na=False)]
    no_action_growth_rows = period_classification[period_classification["verdict"].str.contains("без Action Log", na=False)]
    manager_share = len(manager_rows) / total_rows * 100
    seasonal_share = len(seasonal_rows) / total_rows * 100
    no_action_growth_share = len(no_action_growth_rows) / total_rows * 100
    clean_action = (
        action_impact.dropna(subset=["manager_effect_pp"]).copy()
        if not action_impact.empty and "manager_effect_pp" in action_impact.columns
        else pd.DataFrame()
    )
    calculated_actions = len(clean_action)
    avg_manager_effect = clean_action["manager_effect_pp"].mean() if not clean_action.empty else np.nan
    median_manager_effect = clean_action["manager_effect_pp"].median() if not clean_action.empty else np.nan
    positive_actions_share = (
        (clean_action["manager_effect_pp"] > threshold_percent).mean() * 100
        if not clean_action.empty else np.nan
    )
    worked_gap = np.nan
    if worked_summary is not None and not worked_summary.empty and "worked_group" in worked_summary.columns:
        worked_map = worked_summary.set_index("worked_group")["avg_individual_effect_pp"].to_dict()
        if "Есть Action Logs" in worked_map and "Без Action Logs" in worked_map:
            worked_gap = worked_map["Есть Action Logs"] - worked_map["Без Action Logs"]
    if calculated_actions < 10:
        verdict = "Недостаточно рассчитанных Action Logs для сильного вывода"
        badge = "badge-warn"
        summary = "Модель построена, но действий с корректным окном до/после мало. Главный вывод лучше делать осторожно."
    elif not pd.isna(avg_manager_effect) and avg_manager_effect >= threshold_percent and positive_actions_share >= 45:
        verdict = "Есть признаки роста выше сезонности после менеджерской проработки"
        badge = "badge-good"
        summary = "После Action Logs объекты в среднем улучшаются сильнее, чем похожие объекты в той же локалии. Это аргумент в пользу индивидуального эффекта менеджеров."
    elif seasonal_share >= manager_share * 1.5 and manager_share < 20:
        verdict = "Рост в основном похож на сезонный / общий локальный фон"
        badge = "badge"
        summary = "Большая часть объект-периодов движется около сезонного ожидания и фона похожих объектов. Вклад менеджеров виден точечно, но не доминирует по выборке."
    else:
        verdict = "Смешанная картина: есть и сезонность, и точечный индивидуальный эффект"
        badge = "badge-warn"
        summary = "Часть объектов растёт вместе с локальным фоном, но отдельные объекты после Action Logs показывают рост выше ожидания. Нужен разбор по subject/outcome и наложениям."
    return {
        "verdict": verdict,
        "badge_class": badge,
        "summary": summary,
        "stats": {
            "manager_share": manager_share,
            "seasonal_share": seasonal_share,
            "no_action_growth_share": no_action_growth_share,
            "calculated_actions": calculated_actions,
            "avg_manager_effect": avg_manager_effect,
            "median_manager_effect": median_manager_effect,
            "positive_actions_share": positive_actions_share,
            "worked_gap": worked_gap,
        }
    }


def df_to_html_table(df, max_rows=30):
    if df is None or df.empty:
        return "<p class='muted'>Нет данных для отображения.</p>"
    shown = df.head(max_rows).copy()
    return shown.to_html(index=False, border=0, classes="data-table")


def fig_to_html(fig, include_plotlyjs=False):
    if fig is None:
        return ""
    return pio.to_html(
        fig,
        include_plotlyjs=include_plotlyjs,
        full_html=False,
        config={"displayModeBar": False}
    )


def make_report_html(
    metric_name, graining,
    pre_window_months, lag_months, post_window_months,
    min_months_per_side, overlap_policy, threshold_percent,
    hotel_type_filter, hotel_search,
    selected_outcomes, selected_subjects,
    bookings_df, filtered_monthly, filtered_actions,
    enriched_df, metric_summary, period_classification,
    action_impact, worked_summary,
    correlation_table, corr_individual, corr_efficiency,
    top_hotels, bottom_hotels
):
    total_hotels = enriched_df["hotel_id"].nunique()
    total_actions = len(filtered_actions)
    if action_impact.empty:
        calculated_actions = 0
        excluded_actions = 0
        overlapped_actions = 0
    else:
        calculated_actions = int(action_impact["manager_effect_pp"].notna().sum())
        excluded_actions = int((action_impact["included_in_effect"] == False).sum()) if "included_in_effect" in action_impact.columns else 0
        overlapped_actions = int((action_impact["has_overlap"] == True).sum()) if "has_overlap" in action_impact.columns else 0
    total_metric = enriched_df[metric_name].sum()
    total_expected = enriched_df["expected_value"].sum()
    total_eff = safe_divide(total_metric, total_expected)
    monthly_fig = create_monthly_actual_expected_fig(enriched_df, metric_name, graining)
    seasonality_fig, seasonality_table = create_all_metrics_seasonality_fig(filtered_monthly)
    multi_metric_fig = create_multi_metric_index_fig(filtered_monthly)
    verdict_fig, verdict_counts = create_verdict_fig(period_classification)
    worked_fig = create_worked_fig(worked_summary) if not worked_summary.empty else None
    outcome_fig, outcome_table = create_action_outcome_fig(action_impact)
    report_corr_fig, report_corr_table = create_correlation_heatmap(correlation_table) if correlation_table is not None and not correlation_table.empty else (None, pd.DataFrame())
    clean_action = action_impact.dropna(subset=["manager_effect_pp"]).copy() if not action_impact.empty else pd.DataFrame()
    if clean_action.empty:
        avg_effect = np.nan
        median_effect = np.nan
        positive_share = np.nan
    else:
        avg_effect = clean_action["manager_effect_pp"].mean()
        median_effect = clean_action["manager_effect_pp"].median()
        positive_share = (clean_action["manager_effect_pp"] > 0).mean() * 100
    if verdict_counts.empty:
        main_verdict = "—"
    else:
        main_verdict = verdict_counts.iloc[0]["verdict"]
    report_date = datetime.now().strftime("%d.%m.%Y %H:%M")
    settings_html = f"""
    <ul>
        <li><b>Главная метрика:</b> {html.escape(METRICS[metric_name])}</li>
        <li><b>Разрез динамики:</b> {html.escape(graining)}</li>
        <li><b>Окно Action Log:</b> {pre_window_months} мес. до / месяц действия исключён / лаг {lag_months} мес. / {post_window_months} мес. после</li>
        <li><b>Минимум месяцев с данными:</b> {min_months_per_side} мес. с каждой стороны</li>
        <li><b>Политика наложений:</b> {html.escape(overlap_policy)}</li>
        <li><b>Порог значимого индивидуального эффекта:</b> {threshold_percent}%</li>
        <li><b>Тип объекта:</b> {html.escape(hotel_type_filter)}</li>
        <li><b>Фильтр hotel_id:</b> {html.escape(hotel_search) if hotel_search else "не задан"}</li>
        <li><b>Outcomes:</b> {html.escape(", ".join(selected_outcomes))}</li>
        <li><b>Subjects:</b> {len(selected_subjects)} выбранных значений</li>
    </ul>
    """
    metric_summary_shown = metric_summary.copy()
    metric_summary_shown["actual_sum"] = metric_summary_shown["actual_sum"].round(2)
    metric_summary_shown["expected_sum"] = metric_summary_shown["expected_sum"].round(2)
    metric_summary_shown["efficiency"] = metric_summary_shown["efficiency"].round(3)
    metric_summary_shown["deviation_pct"] = (metric_summary_shown["deviation_pct"] * 100).round(2)
    metric_summary_shown = metric_summary_shown[[
        "metric_label", "actual_sum", "expected_sum",
        "efficiency", "deviation_pct", "description"
    ]]
    period_table = period_classification.copy()
    for col in ["actual_value", "expected_value"]:
        period_table[col] = period_table[col].round(2)
    for col in ["hotel_efficiency", "segment_efficiency"]:
        period_table[col] = period_table[col].round(3)
    for col in ["hotel_residual_pp", "segment_residual_pp", "individual_effect_pp"]:
        period_table[col] = period_table[col].round(2)
    period_table = period_table[[
        "hotel_id", "period_label", "segment_key",
        "actual_value", "expected_value", "hotel_efficiency",
        "individual_effect_pp", "actions_count", "verdict"
    ]]
    html_parts = f"""
    <!doctype html>
    <html lang="ru">
    <head>
        <meta charset="utf-8">
        <title>Hotel Seasonality & Manager Impact Report</title>
        <style>
            body {{
                font-family: Inter, Arial, sans-serif;
                margin: 0;
                background:
                    radial-gradient(circle at 8% 6%, rgba(34, 211, 238, 0.22), transparent 30%),
                    radial-gradient(circle at 82% 10%, rgba(168, 85, 247, 0.22), transparent 32%),
                    radial-gradient(circle at 12% 86%, rgba(244, 114, 182, 0.13), transparent 30%),
                    linear-gradient(135deg, #020617 0%, #08111f 42%, #111827 100%);
                color: #eef2ff;
            }}
            .page {{ max-width: 1180px; margin: 0 auto; padding: 36px 28px 60px; }}
            .hero {{
                background:
                    radial-gradient(circle at 12% 18%, rgba(34, 211, 238, 0.40), transparent 30%),
                    radial-gradient(circle at 76% 10%, rgba(168, 85, 247, 0.36), transparent 32%),
                    linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.78));
                color: white; border: 1px solid rgba(255, 255, 255, 0.16);
                border-radius: 26px; padding: 34px 38px; margin-bottom: 24px;
                box-shadow: 0 28px 85px rgba(0, 0, 0, 0.38);
            }}
            .hero h1 {{ margin: 0 0 10px; font-size: 34px; }}
            .hero p {{ margin: 0; font-size: 16px; color: #dbeafe; }}
            .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 20px 0; }}
            .card {{
                background:
                    linear-gradient(145deg, rgba(15, 23, 42, 0.84), rgba(30, 41, 59, 0.60));
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-radius: 20px; padding: 18px 20px;
                box-shadow: 0 18px 55px rgba(0, 0, 0, 0.28); margin-bottom: 18px;
            }}
            .metric-title {{ font-size: 13px; color: #cbd5e1; margin-bottom: 6px; }}
            .metric-value {{ font-size: 25px; font-weight: 800; color: #f8fafc; }}
            h2 {{ margin-top: 34px; margin-bottom: 12px; font-size: 24px; color: #f8fafc; }}
            h3 {{ margin-top: 18px; font-size: 18px; color: #e0f2fe; }}
            .muted {{ color: #94a3b8; }}
            .data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: rgba(15, 23, 42, 0.72); color: #e5eefb; }}
            .data-table th {{ text-align: left; background: rgba(30, 41, 59, 0.92); color: #f8fafc; padding: 9px; border-bottom: 1px solid rgba(148, 163, 184, 0.28); }}
            .data-table td {{ padding: 8px 9px; border-bottom: 1px solid rgba(148, 163, 184, 0.16); }}
            .note {{ background: rgba(14, 165, 233, 0.12); border-left: 5px solid #38bdf8; padding: 14px 16px; border-radius: 12px; color: #e0f2fe; }}
            .formula {{ font-family: Consolas, monospace; background: rgba(2, 6, 23, 0.88); color: #dbeafe; padding: 12px 14px; border-radius: 12px; overflow-x: auto; border: 1px solid rgba(148, 163, 184, 0.22); }}
        </style>
    </head>
    <body>
    <div class="page">
        <div class="hero">
            <h1>Анализ сезонности и влияния Action Logs</h1>
            <p>Отчёт сформирован по текущим фильтрам дашборда: {report_date}</p>
        </div>
        <div class="note">
            <b>Цель:</b> отделить естественный сезонный рост одной локалии от индивидуального эффекта объекта и возможного вклада менеджерской проработки.
            Региональное разделение не используется: вся выборка считается одной локалией, а сравнение аналогичных объектов идёт только внутри типа STR / HOTEL.
        </div>
        <h2>1. Настройки выборки</h2>
        <div class="card">{settings_html}</div>
        <h2>2. Ключевые показатели</h2>
        <div class="grid">
            <div class="card"><div class="metric-title">Отелей в текущей выборке</div><div class="metric-value">{format_number(total_hotels)}</div></div>
            <div class="card"><div class="metric-title">Action Logs после фильтров</div><div class="metric-value">{format_number(total_actions)}</div></div>
            <div class="card"><div class="metric-title">Action Logs с рассчитанным эффектом</div><div class="metric-value">{format_number(calculated_actions)}</div></div>
            <div class="card"><div class="metric-title">Факт по главной метрике</div><div class="metric-value">{format_number(total_metric)}</div></div>
            <div class="card"><div class="metric-title">Факт / ожидание</div><div class="metric-value">{format_number(total_eff, 3)}</div></div>
        </div>
        <h2>3. Что показывает модель</h2>
        <div class="card">
            <p>Основной эффект считается по выбранной главной метрике: {html.escape(METRICS[metric_name])}. Остальные показатели используются для сравнения.</p>
            <p class="formula">expected = hotel_baseline × seasonality_index</p>
            <p class="formula">individual_effect = hotel_residual - segment_residual</p>
            <p class="formula">manager_effect = (post_hotel_eff - pre_hotel_eff) - (post_peer_eff - pre_peer_eff)</p>
            <p class="muted">Segment в отчёте – это не регион, а тип объекта: STR / HOTEL. Вся локалия одна.</p>
            <p><b>Окно оценки Action Log:</b> {pre_window_months} мес. до, месяц действия исключён, лаг {lag_months} мес., {post_window_months} мес. после.</p>
            <p><b>Достаточность данных:</b> минимум {min_months_per_side} месяцев с каждой стороны. Рассчитано действий: {format_number(calculated_actions)}, исключено/не хватило данных: {format_number(excluded_actions)}, с наложениями: {format_number(overlapped_actions)}.</p>
            <p class="muted">Если у одного отеля несколько Action Logs попадают в одно окно, эффект считается потенциально смешанным. Политика текущего отчёта: {html.escape(overlap_policy)}.</p>
        </div>
        <h2>4. Итог по дополнительным метрикам</h2>
        <div class="card">{df_to_html_table(metric_summary_shown, max_rows=20)}</div>
        <h2>5. Графики</h2>
        <div class="card">{fig_to_html(monthly_fig, include_plotlyjs=True)}</div>
        <div class="card">{fig_to_html(seasonality_fig, include_plotlyjs=False)}</div>
        <div class="card">{fig_to_html(multi_metric_fig, include_plotlyjs=False)}</div>
        <div class="card">{fig_to_html(verdict_fig, include_plotlyjs=False)}</div>
    """
    if worked_fig is not None:
        html_parts += f"<div class='card'>{fig_to_html(worked_fig, include_plotlyjs=False)}</div>"
    if outcome_fig is not None:
        html_parts += f"<div class='card'>{fig_to_html(outcome_fig, include_plotlyjs=False)}</div>"
    if report_corr_fig is not None:
        html_parts += f"<h2>5.1. Матрица корреляций</h2><div class='card'>{fig_to_html(report_corr_fig, include_plotlyjs=False)}</div>"
    html_parts += f"""
        <h2>6. Главный ответ и выводы по текущей выборке</h2>
        <div class="card">
            <p><b>Главный вопрос:</b> рост оценивается как сезонность/локальный фон или рост выше фона после менеджерской проработки.</p>
            <p><b>Самая частая классификация объект-периода:</b> {html.escape(str(main_verdict))}</p>
            <p><b>Средний manager effect:</b> {format_percent(avg_effect, 2)} п.п.</p>
            <p><b>Медианный manager effect:</b> {format_percent(median_effect, 2)} п.п.</p>
            <p><b>Доля положительных Action Logs:</b> {format_percent(positive_share, 1)}</p>
            <p><b>Корреляция действий и индивидуального эффекта:</b> {format_number(corr_individual, 3)}</p>
            <p><b>Корреляция действий и эффективности отеля:</b> {format_number(corr_efficiency, 3)}</p>
            <p class="muted">Корреляция не доказывает причинность, но показывает направление связи между количеством действий и ростом относительно сезонного ожидания.</p>
        </div>
        <h2>7. Таблицы для защиты</h2>
        <h3>Топ объектов относительно сезонного ожидания</h3>
        <div class="card">{df_to_html_table(top_hotels, max_rows=20)}</div>
        <h3>Слабые объекты относительно сезонного ожидания</h3>
        <div class="card">{df_to_html_table(bottom_hotels, max_rows=20)}</div>
        <h3>Классификация объект-периодов</h3>
        <div class="card">{df_to_html_table(period_table, max_rows=50)}</div>
        <h3>Эффект по outcome</h3>
        <div class="card">{df_to_html_table(outcome_table.round(2) if not outcome_table.empty else outcome_table, max_rows=20)}</div>
        <h2>8. Ограничения</h2>
        <div class="card">
            <ul>
                <li>Модель показывает вероятный вклад менеджерской работы, а не абсолютное доказательство причинности.</li>
                <li>Месяц самого действия не входит в before/after окно, чтобы не смешивать частичный эффект.</li>
                <li>Revenue объекта не равен комиссии платформы.</li>
                <li>Если у объекта мало активных месяцев, его рейтинг менее надёжен.</li>
            </ul>
        </div>
    </div>
    </body>
    </html>
    """
    return html_parts
