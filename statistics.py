import math

import numpy as np
import pandas as pd

from config import SCIPY_AVAILABLE, scipy_stats, SUCCESS_OUTCOMES, NEGATIVE_OUTCOMES


def _normal_two_sided_pvalue_from_t(t_value):
    if pd.isna(t_value):
        return np.nan
    z = abs(float(t_value))
    return 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))


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
            p_value = float(_normal_two_sided_pvalue_from_t(t_stat))
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
            p_value = float(_normal_two_sided_pvalue_from_t(t_stat))
    return mean_a, mean_b, mean_a - mean_b, t_stat, n_a, n_b, df, p_value, math.sqrt(se2) if se2 > 0 else np.nan, "scipy" if SCIPY_AVAILABLE else "normal approx"


def format_pvalue(p):
    if pd.isna(p):
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


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
