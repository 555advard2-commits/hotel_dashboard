import numpy as np
import pandas as pd
import plotly.express as px

from config import METRICS
from utils import safe_divide


def calculate_all_metric_hotel_features(monthly_df, period_classification=None, action_impact=None):
    if monthly_df.empty:
        return pd.DataFrame()
    rows = []
    for hotel_id, g in monthly_df.groupby("hotel_id"):
        row = {
            "hotel_id": str(hotel_id),
            "is_STR": bool(g["is_STR"].iloc[0]) if "is_STR" in g.columns else False,
            "months_count": int(g["month"].nunique()) if "month" in g.columns else len(g),
        }
        for m in METRICS:
            vals = pd.to_numeric(g[m], errors="coerce").fillna(0)
            row[f"{m}_sum"] = float(vals.sum())
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_std"] = float(vals.std(ddof=0))
            row[f"{m}_zero_share"] = float((vals <= 0).mean() * 100)
            row[f"{m}_cv"] = safe_divide(float(vals.std(ddof=0)), float(vals.mean())) if vals.mean() != 0 else np.nan
        row["avg_roomnights_per_booking"] = safe_divide(row.get("roomnights_sum", 0), row.get("gbb_sum", 0))
        row["avg_sales_per_booking"] = safe_divide(row.get("sales_volumes_rub_sum", 0), row.get("gbb_sum", 0))
        row["avg_sales_per_roomnight"] = safe_divide(row.get("sales_volumes_rub_sum", 0), row.get("roomnights_sum", 0))
        row["revenue_to_sales_pct"] = safe_divide(row.get("revenue_rub_sum", 0), row.get("sales_volumes_rub_sum", 0)) * 100
        rows.append(row)
    features = pd.DataFrame(rows)
    if period_classification is not None and not period_classification.empty:
        pc = period_classification.copy()
        agg = pc.groupby("hotel_id", as_index=False).agg(
            avg_individual_effect_pp=("individual_effect_pp", "mean"),
            median_individual_effect_pp=("individual_effect_pp", "median"),
            avg_hotel_efficiency=("hotel_efficiency", "mean"),
            object_periods=("hotel_id", "count"),
            periods_with_actions=("actions_count", lambda x: int((x > 0).sum()))
        )
        features = features.merge(agg, on="hotel_id", how="left")
    if action_impact is not None and not action_impact.empty and "manager_effect_pp" in action_impact.columns:
        ai = action_impact.copy()
        agg_ai = ai.groupby("hotel_id", as_index=False).agg(
            actions_total=("hotel_id", "count"),
            actions_with_effect=("manager_effect_pp", lambda x: int(pd.Series(x).notna().sum())),
            avg_manager_effect_pp=("manager_effect_pp", "mean"),
            median_manager_effect_pp=("manager_effect_pp", "median"),
            overlap_actions=("has_overlap", lambda x: int(pd.Series(x).fillna(False).sum()) if len(x) else 0)
        )
        features = features.merge(agg_ai, on="hotel_id", how="left")
    for col in [
        "avg_individual_effect_pp", "median_individual_effect_pp", "avg_hotel_efficiency",
        "object_periods", "periods_with_actions", "actions_total", "actions_with_effect",
        "avg_manager_effect_pp", "median_manager_effect_pp", "overlap_actions"
    ]:
        if col in features.columns:
            features[col] = pd.to_numeric(features[col], errors="coerce")
    return features


def weighted_standardize_features(features_df, feature_columns, weight_map):
    x = features_df[feature_columns].copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    medians = x.median(numeric_only=True)
    x = x.fillna(medians).fillna(0)
    means = x.mean(axis=0)
    stds = x.std(axis=0, ddof=0).replace(0, 1)
    z = (x - means) / stds
    for col in z.columns:
        weight = weight_map.get(col, 1.0)
        z[col] = z[col] * weight
    return z.astype(float), means, stds


def simple_kmeans(matrix, k=5, max_iter=100, random_state=42):
    x = np.asarray(matrix, dtype=float)
    n = x.shape[0]
    if n == 0 or k < 1:
        return np.array([]), np.empty((0, x.shape[1] if x.ndim == 2 else 0)), np.nan
    k = min(k, n)
    rng = np.random.default_rng(random_state)
    norms = np.linalg.norm(x, axis=1)
    order = np.argsort(norms)
    init_idx = np.linspace(0, n - 1, k).round().astype(int)
    centers = x[order[init_idx]].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for cluster_id in range(k):
            mask = labels == cluster_id
            if mask.any():
                centers[cluster_id] = x[mask].mean(axis=0)
            else:
                centers[cluster_id] = x[rng.integers(0, n)]
    inertia = float(((x - centers[labels]) ** 2).sum())
    return labels, centers, inertia


def add_pca_projection(weighted_matrix):
    x = np.asarray(weighted_matrix, dtype=float)
    if x.shape[0] < 2 or x.shape[1] < 2:
        return np.zeros(x.shape[0]), np.zeros(x.shape[0]), np.nan, np.nan
    x_centered = x - x.mean(axis=0)
    try:
        u, s, vt = np.linalg.svd(x_centered, full_matrices=False)
        coords = u[:, :2] * s[:2]
        total_var = (s ** 2).sum()
        evr1 = (s[0] ** 2 / total_var) if total_var > 0 else np.nan
        evr2 = (s[1] ** 2 / total_var) if len(s) > 1 and total_var > 0 else np.nan
        return coords[:, 0], coords[:, 1], evr1, evr2
    except Exception:
        return np.zeros(x.shape[0]), np.zeros(x.shape[0]), np.nan, np.nan


def build_cluster_analysis(monthly_df, period_classification, action_impact, weights, k=5):
    features = calculate_all_metric_hotel_features(monthly_df, period_classification, action_impact)
    if features.empty:
        return pd.DataFrame(), pd.DataFrame(), None, None, pd.DataFrame()
    feature_groups = {
        "gbb": ["gbb_sum", "gbb_mean", "gbb_std", "gbb_zero_share", "gbb_cv"],
        "roomnights": [
            "roomnights_sum", "roomnights_mean", "roomnights_std",
            "roomnights_zero_share", "roomnights_cv", "avg_roomnights_per_booking"
        ],
        "sales_volumes_rub": [
            "sales_volumes_rub_sum", "sales_volumes_rub_mean", "sales_volumes_rub_std",
            "sales_volumes_rub_zero_share", "sales_volumes_rub_cv",
            "avg_sales_per_booking", "avg_sales_per_roomnight"
        ],
        "revenue_rub": [
            "revenue_rub_sum", "revenue_rub_mean", "revenue_rub_std",
            "revenue_rub_zero_share", "revenue_rub_cv", "revenue_to_sales_pct"
        ],
        "manager": [
            "avg_individual_effect_pp", "avg_hotel_efficiency",
            "actions_total", "avg_manager_effect_pp", "overlap_actions"
        ]
    }
    feature_columns = []
    weight_map = {}
    for group_name, cols in feature_groups.items():
        group_weight = weights.get(group_name, 1.0)
        for col in cols:
            if col in features.columns:
                feature_columns.append(col)
                weight_map[col] = group_weight
    usable = []
    for col in feature_columns:
        series = pd.to_numeric(features[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if series.notna().sum() >= 3 and series.nunique(dropna=True) > 1:
            usable.append(col)
    feature_columns = usable
    if len(feature_columns) < 2 or len(features) < 3:
        return features, pd.DataFrame(), None, None, pd.DataFrame()
    weighted_z, means, stds = weighted_standardize_features(features, feature_columns, weight_map)
    labels, centers, inertia = simple_kmeans(weighted_z.values, k=k, max_iter=120, random_state=42)
    features = features.copy()
    features["cluster"] = labels + 1
    features["cluster_label"] = features["cluster"].apply(lambda x: f"Кластер {int(x)}")
    pc1, pc2, evr1, evr2 = add_pca_projection(weighted_z.values)
    features["map_x"] = pc1
    features["map_y"] = pc2
    summary = features.groupby("cluster_label", as_index=False).agg(
        hotels_count=("hotel_id", "count"),
        gbb_sum=("gbb_sum", "sum"),
        avg_gbb_per_month=("gbb_mean", "mean"),
        avg_roomnights_per_month=("roomnights_mean", "mean"),
        avg_sales_per_month=("sales_volumes_rub_mean", "mean"),
        avg_revenue_per_month=("revenue_rub_mean", "mean"),
        avg_sales_per_booking=("avg_sales_per_booking", "mean"),
        avg_individual_effect_pp=("avg_individual_effect_pp", "mean"),
        avg_manager_effect_pp=("avg_manager_effect_pp", "mean"),
        avg_actions_total=("actions_total", "mean")
    )
    summary = summary.sort_values("hotels_count", ascending=False)
    fig_scatter = px.scatter(
        features,
        x="map_x", y="map_y", color="cluster_label",
        hover_data=[
            "hotel_id", "is_STR", "gbb_sum", "roomnights_sum",
            "sales_volumes_rub_sum", "revenue_rub_sum",
            "avg_individual_effect_pp", "avg_manager_effect_pp"
        ],
        labels={
            "map_x": f"Компонента 1 ({evr1*100:.1f}% разброса)" if not pd.isna(evr1) else "Компонента 1",
            "map_y": f"Компонента 2 ({evr2*100:.1f}% разброса)" if not pd.isna(evr2) else "Компонента 2",
            "cluster_label": "Кластер"
        },
        title="Карта отелей по всем метрикам: взвешенная кластеризация"
    )
    fig_scatter.update_layout(height=620)
    profile_cols = [
        "cluster_label", "hotels_count", "avg_gbb_per_month",
        "avg_roomnights_per_month", "avg_sales_per_month",
        "avg_revenue_per_month", "avg_individual_effect_pp",
        "avg_manager_effect_pp", "avg_actions_total"
    ]
    heat = summary[profile_cols].copy().set_index("cluster_label")
    heat_num = heat.replace([np.inf, -np.inf], np.nan).fillna(0)
    heat_z = (heat_num - heat_num.mean()) / heat_num.std(ddof=0).replace(0, 1)
    fig_profile = px.imshow(
        heat_z.T, text_auto=".1f", aspect="auto",
        color_continuous_scale=["#59758C", "#FFFFFF", "#CC7D5E"],
        labels={"color": "Z-score"},
        title="Профиль кластеров: какие признаки выше/ниже среднего"
    )
    fig_profile.update_layout(height=620, margin=dict(l=20, r=20, t=70, b=20))
    weight_table = pd.DataFrame([
        {"Группа признаков": "GBB / бронирования", "Вес": weights.get("gbb", 1.0), "Что усиливает": "масштаб и стабильность бронирований"},
        {"Группа признаков": "Roomnights / ночи", "Вес": weights.get("roomnights", 1.0), "Что усиливает": "длина проживания и глубина спроса"},
        {"Группа признаков": "Sales volume / объём продаж", "Вес": weights.get("sales_volumes_rub", 1.0), "Что усиливает": "денежный масштаб объекта"},
        {"Группа признаков": "Revenue объекта", "Вес": weights.get("revenue_rub", 1.0), "Что усиливает": "доход объекта"},
        {"Группа признаков": "Manager / Action Logs", "Вес": weights.get("manager", 1.0), "Что усиливает": "проработку, individual effect и manager effect"},
    ])
    return features, summary, fig_scatter, fig_profile, weight_table
