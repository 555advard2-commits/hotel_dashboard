import numpy as np
import pandas as pd

from utils import parse_bool, parse_number, safe_divide


MONTH_NAMES_SHORT = {
    1: "Янв", 2: "Фев", 3: "Мар", 4: "Апр", 5: "Май", 6: "Июн",
    7: "Июл", 8: "Авг", 9: "Сен", 10: "Окт", 11: "Ноя", 12: "Дек"
}


def build_monthly_panel(bookings_df):
    """
    Собирает месячную панель hotel_id x month.
    """
    if bookings_df is None or bookings_df.empty:
        return pd.DataFrame()
    df = bookings_df.copy()
    if "month" not in df.columns:
        df["hotel_id"] = df["hotel_id"].astype(str).str.strip()
        df["booking_created_date"] = pd.to_datetime(df["booking_created_date"], errors="coerce")
        df = df.dropna(subset=["hotel_id", "booking_created_date"])
        df["month"] = df["booking_created_date"].dt.to_period("M").dt.to_timestamp()
        df["month_num"] = df["month"].dt.month
        df["is_STR"] = df["is_STR"].apply(parse_bool)
        for col in ["gbb", "roomnights", "sales_volumes_rub", "revenue_rub"]:
            df[col] = df[col].apply(parse_number)
        grouped = (
            df.groupby(["hotel_id", "month", "month_num"], as_index=False)
            .agg(
                is_STR=("is_STR", lambda x: bool(pd.Series(x).mode().iloc[0]) if not pd.Series(x).mode().empty else False),
                gbb_month=("gbb", "sum"),
                roomnights_month=("roomnights", "sum"),
                sales_month=("sales_volumes_rub", "sum"),
                revenue_month=("revenue_rub", "sum"),
                source_rows=("hotel_id", "size"),
            )
        )
        grouped["has_source_row"] = grouped["source_rows"] > 0
        grouped = grouped.drop(columns=["source_rows"])
    else:
        rename_map = {
            "gbb": "gbb_month",
            "roomnights": "roomnights_month",
            "sales_volumes_rub": "sales_month",
            "revenue_rub": "revenue_month",
        }
        df["hotel_id"] = df["hotel_id"].astype(str).str.strip()
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
        if "month_num" not in df.columns:
            df["month_num"] = df["month"].dt.month
        grouped = df.rename(columns=rename_map)
        for col in ["gbb_month", "roomnights_month", "sales_month", "revenue_month"]:
            grouped[col] = pd.to_numeric(grouped[col], errors="coerce").fillna(0)
        grouped["is_STR"] = grouped["is_STR"].apply(parse_bool) if "is_STR" in grouped.columns else False
        keep_cols = ["hotel_id", "month", "month_num", "is_STR", "gbb_month", "roomnights_month", "sales_month", "revenue_month"]
        if "has_source_row" in grouped.columns:
            keep_cols.append("has_source_row")
        grouped = grouped[keep_cols].copy()
    grouped = grouped.dropna(subset=["hotel_id", "month"])
    if "has_source_row" in grouped.columns:
        grouped["has_source_row"] = grouped["has_source_row"].fillna(False).astype(bool)
    else:
        grouped["has_source_row"] = (
            grouped["gbb_month"].ne(0)
            | grouped["roomnights_month"].ne(0)
            | grouped["sales_month"].ne(0)
            | grouped["revenue_month"].ne(0)
        )
    return grouped.sort_values(["hotel_id", "month"]).reset_index(drop=True)


def _safe_ratio(numerator, denominator):
    return safe_divide(float(numerator), float(denominator)) if denominator and denominator > 0 else np.nan


def _growth_2025_vs_2024(group, value_col):
    year_totals = group.groupby(group["month"].dt.year)[value_col].sum()
    y2024 = float(year_totals.get(2024, 0))
    y2025 = float(year_totals.get(2025, 0))
    return _safe_ratio(y2025, y2024) - 1 if y2024 > 0 else np.nan


def _median_monthly_yoy(group, value_col):
    pivot = group.pivot_table(index="month_num", columns=group["month"].dt.year, values=value_col, aggfunc="sum")
    if 2024 not in pivot.columns or 2025 not in pivot.columns:
        return np.nan
    ratios = []
    for _, row in pivot.iterrows():
        base = row.get(2024, 0)
        current = row.get(2025, 0)
        if pd.notna(base) and base > 0 and pd.notna(current):
            ratios.append((current / base) - 1)
    return float(np.median(ratios)) if ratios else np.nan


def _seasonality_stats(group, value_col, prefix):
    values = pd.to_numeric(group[value_col], errors="coerce").fillna(0)
    total = float(values.sum())
    monthly_by_calendar = group.groupby("month_num")[value_col].sum().reindex(range(1, 13), fill_value=0)
    shares = monthly_by_calendar / total if total > 0 else monthly_by_calendar * 0
    active_values = values[values > 0]
    avg = float(values.mean()) if len(values) else 0
    std = float(values.std(ddof=0)) if len(values) else 0
    top3_share = float(np.sort(values.values)[-3:].sum() / total) if total > 0 and len(values) else np.nan
    peak_month = int(monthly_by_calendar.idxmax()) if total > 0 else np.nan
    row = {
        f"{prefix}_SI": _safe_ratio(float(monthly_by_calendar.max()), float(monthly_by_calendar.mean())),
        f"{prefix}_CV": _safe_ratio(std, avg) if avg > 0 else np.nan,
        f"{prefix}_top3_share": top3_share,
        f"{prefix}_summer_share": float(shares.loc[[6, 7, 8]].sum()) if total > 0 else np.nan,
        f"{prefix}_winter_share": float(shares.loc[[12, 1, 2]].sum()) if total > 0 else np.nan,
        f"{prefix}_peak_month": peak_month,
        f"{prefix}_active_months": int(len(active_values)),
    }
    for month_num in range(1, 13):
        row[f"{prefix}_share_m{month_num:02d}"] = float(shares.loc[month_num]) if total > 0 else 0.0
    return row


def build_hotel_features(monthly_panel):
    """
    Считает признаки на уровне одного hotel_id.
    """
    if monthly_panel is None or monthly_panel.empty:
        return pd.DataFrame()
    df = monthly_panel.copy()
    df["hotel_id"] = df["hotel_id"].astype(str).str.strip()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df.dropna(subset=["hotel_id", "month"])
    if df.empty:
        return pd.DataFrame()
    if "month_num" not in df.columns:
        df["month_num"] = df["month"].dt.month
    if "has_source_row" not in df.columns:
        df["has_source_row"] = True
    if "is_STR" not in df.columns:
        df["is_STR"] = False
    for col in ["gbb_month", "roomnights_month", "sales_month", "revenue_month"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["has_source_row"] = df["has_source_row"].fillna(False).astype(bool)
    df["is_STR"] = df["is_STR"].apply(parse_bool)
    df["year"] = df["month"].dt.year
    df["active_row"] = df[["gbb_month", "roomnights_month", "sales_month", "revenue_month"]].sum(axis=1) > 0

    grouped = df.groupby("hotel_id", sort=False)
    features = grouped.agg(
        is_STR=("is_STR", "first"),
        months_available=("has_source_row", "sum"),
        total_months=("month", "nunique"),
        active_months=("active_row", "sum"),
        revenue_sum=("revenue_month", "sum"),
        sales_sum=("sales_month", "sum"),
        roomnights_sum=("roomnights_month", "sum"),
        gbb_sum=("gbb_month", "sum"),
    )
    features["months_available"] = features["months_available"].astype(int)
    features["active_months"] = features["active_months"].astype(int)
    features["zero_months"] = (features["total_months"] - features["active_months"]).clip(lower=0).astype(int)
    features["has_full_24m"] = features["months_available"] >= 24

    denom = features["months_available"].replace(0, np.nan)
    features["avg_monthly_revenue"] = features["revenue_sum"] / denom
    features["avg_monthly_sales"] = features["sales_sum"] / denom
    features["avg_monthly_roomnights"] = features["roomnights_sum"] / denom
    features["avg_monthly_gbb"] = features["gbb_sum"] / denom
    features["total_revenue_24m"] = features["revenue_sum"]
    features["ADR"] = features["revenue_sum"] / features["roomnights_sum"].replace(0, np.nan)
    features["sales_per_roomnight"] = features["sales_sum"] / features["roomnights_sum"].replace(0, np.nan)
    features["revenue_per_booking"] = features["revenue_sum"] / features["gbb_sum"].replace(0, np.nan)
    features["LOS"] = features["roomnights_sum"] / features["gbb_sum"].replace(0, np.nan)

    for source, target in [
        ("avg_monthly_revenue", "log_avg_revenue"),
        ("avg_monthly_sales", "log_avg_sales"),
        ("total_revenue_24m", "log_total_revenue"),
        ("avg_monthly_roomnights", "log_avg_roomnights"),
        ("avg_monthly_gbb", "log_avg_gbb"),
    ]:
        features[target] = np.log1p(features[source].clip(lower=0))

    for value_col, growth_col in [
        ("revenue_month", "growth_revenue_yoy"),
        ("roomnights_month", "growth_roomnights_yoy"),
        ("gbb_month", "growth_gbb_yoy"),
    ]:
        yearly = df.groupby(["hotel_id", "year"], sort=False)[value_col].sum().unstack(fill_value=0)
        y2024 = yearly[2024] if 2024 in yearly.columns else pd.Series(0.0, index=yearly.index)
        y2025 = yearly[2025] if 2025 in yearly.columns else pd.Series(0.0, index=yearly.index)
        features[growth_col] = (y2025 / y2024.replace(0, np.nan)) - 1

    monthly_yoy = (
        df.groupby(["hotel_id", "month_num", "year"], sort=False)["revenue_month"]
        .sum()
        .unstack(fill_value=0)
    )
    if 2024 in monthly_yoy.columns and 2025 in monthly_yoy.columns:
        yoy_ratio = (monthly_yoy[2025] / monthly_yoy[2024].replace(0, np.nan)) - 1
        features["median_monthly_yoy"] = yoy_ratio.groupby(level=0).median()
    else:
        features["median_monthly_yoy"] = np.nan

    def add_seasonality(value_col, prefix):
        calendar = (
            df.groupby(["hotel_id", "month_num"], sort=False)[value_col]
            .sum()
            .unstack(fill_value=0)
            .reindex(columns=range(1, 13), fill_value=0)
        )
        month_matrix = (
            df.pivot_table(index="hotel_id", columns="month", values=value_col, aggfunc="sum", fill_value=0)
            .reindex(features.index, fill_value=0)
        )
        totals = calendar.sum(axis=1)
        shares = calendar.div(totals.replace(0, np.nan), axis=0).fillna(0)
        mean_values = month_matrix.mean(axis=1)
        std_values = month_matrix.std(axis=1, ddof=0)
        sorted_values = np.sort(month_matrix.to_numpy(dtype=float), axis=1)
        top3 = sorted_values[:, -3:].sum(axis=1)
        features[f"{prefix}_SI"] = calendar.max(axis=1) / calendar.mean(axis=1).replace(0, np.nan)
        features[f"{prefix}_CV"] = std_values / mean_values.replace(0, np.nan)
        features[f"{prefix}_top3_share"] = pd.Series(top3, index=month_matrix.index) / totals.replace(0, np.nan)
        features[f"{prefix}_summer_share"] = shares[[6, 7, 8]].sum(axis=1)
        features[f"{prefix}_winter_share"] = shares[[12, 1, 2]].sum(axis=1)
        features[f"{prefix}_peak_month"] = calendar.idxmax(axis=1).where(totals > 0, np.nan)
        features[f"{prefix}_active_months"] = (month_matrix > 0).sum(axis=1)
        for month_num in range(1, 13):
            features[f"{prefix}_share_m{month_num:02d}"] = shares[month_num]

    add_seasonality("roomnights_month", "roomnights")
    add_seasonality("gbb_month", "gbb")
    add_seasonality("revenue_month", "revenue")

    features["SI"] = features["roomnights_SI"]
    features["CV"] = features["roomnights_CV"]
    features["top3_share"] = features["roomnights_top3_share"]
    features["summer_share"] = features["roomnights_summer_share"]
    features["winter_share"] = features["roomnights_winter_share"]
    features["peak_month"] = features["roomnights_peak_month"]
    features["growth_2025_vs_2024"] = features["growth_revenue_yoy"]
    features = features.replace([np.inf, -np.inf], np.nan).reset_index()
    return features.drop(columns=["total_months", "revenue_sum", "sales_sum", "roomnights_sum", "gbb_sum"], errors="ignore")
    rows = []
    for hotel_id, group in monthly_panel.groupby("hotel_id", sort=False):
        group = group.sort_values("month").copy()
        months_available = int(group["has_source_row"].fillna(False).sum()) if "has_source_row" in group.columns else int(group["month"].nunique())
        active_months = int((group[["gbb_month", "roomnights_month", "sales_month", "revenue_month"]].sum(axis=1) > 0).sum())
        zero_months = int(max(group["month"].nunique() - active_months, 0))
        revenue_sum = float(group["revenue_month"].sum())
        sales_sum = float(group["sales_month"].sum())
        roomnights_sum = float(group["roomnights_month"].sum())
        gbb_sum = float(group["gbb_month"].sum())
        row = {
            "hotel_id": str(hotel_id),
            "is_STR": bool(group["is_STR"].iloc[0]) if "is_STR" in group.columns else False,
            "months_available": months_available,
            "active_months": active_months,
            "zero_months": zero_months,
            "has_full_24m": bool(months_available >= 24),
            "avg_monthly_revenue": revenue_sum / months_available if months_available else np.nan,
            "avg_monthly_sales": sales_sum / months_available if months_available else np.nan,
            "avg_monthly_roomnights": roomnights_sum / months_available if months_available else np.nan,
            "avg_monthly_gbb": gbb_sum / months_available if months_available else np.nan,
            "total_revenue_24m": revenue_sum,
            "ADR": _safe_ratio(revenue_sum, roomnights_sum),
            "sales_per_roomnight": _safe_ratio(sales_sum, roomnights_sum),
            "revenue_per_booking": _safe_ratio(revenue_sum, gbb_sum),
            "LOS": _safe_ratio(roomnights_sum, gbb_sum),
            "growth_revenue_yoy": _growth_2025_vs_2024(group, "revenue_month"),
            "growth_roomnights_yoy": _growth_2025_vs_2024(group, "roomnights_month"),
            "growth_gbb_yoy": _growth_2025_vs_2024(group, "gbb_month"),
            "median_monthly_yoy": _median_monthly_yoy(group, "revenue_month"),
        }
        row["log_avg_revenue"] = np.log1p(max(row["avg_monthly_revenue"], 0)) if pd.notna(row["avg_monthly_revenue"]) else np.nan
        row["log_avg_sales"] = np.log1p(max(row["avg_monthly_sales"], 0)) if pd.notna(row["avg_monthly_sales"]) else np.nan
        row["log_total_revenue"] = np.log1p(max(row["total_revenue_24m"], 0)) if pd.notna(row["total_revenue_24m"]) else np.nan
        row["log_avg_roomnights"] = np.log1p(max(row["avg_monthly_roomnights"], 0)) if pd.notna(row["avg_monthly_roomnights"]) else np.nan
        row["log_avg_gbb"] = np.log1p(max(row["avg_monthly_gbb"], 0)) if pd.notna(row["avg_monthly_gbb"]) else np.nan
        row.update(_seasonality_stats(group, "roomnights_month", "roomnights"))
        row.update(_seasonality_stats(group, "gbb_month", "gbb"))
        row.update(_seasonality_stats(group, "revenue_month", "revenue"))
        row["SI"] = row["roomnights_SI"]
        row["CV"] = row["roomnights_CV"]
        row["top3_share"] = row["roomnights_top3_share"]
        row["summer_share"] = row["roomnights_summer_share"]
        row["winter_share"] = row["roomnights_winter_share"]
        row["peak_month"] = row["roomnights_peak_month"]
        row["growth_2025_vs_2024"] = row["growth_revenue_yoy"]
        rows.append(row)
    return pd.DataFrame(rows)


def _standardize(features_df, feature_columns):
    x = features_df[feature_columns].replace([np.inf, -np.inf], np.nan).copy()
    medians = x.median(numeric_only=True)
    x = x.fillna(medians).fillna(0)
    means = x.mean(axis=0)
    stds = x.std(axis=0, ddof=0).replace(0, 1)
    scaled = (x - means) / stds
    return scaled.astype(float), means, stds


def _simple_kmeans(matrix, k=5, max_iter=60, random_state=42, tol=1e-4):
    x = np.asarray(matrix, dtype=float)
    n = x.shape[0]
    if n == 0:
        return np.array([]), np.empty((0, x.shape[1] if x.ndim == 2 else 0)), np.nan
    k = max(1, min(int(k), n))
    order = np.argsort(np.linalg.norm(x, axis=1))
    init_idx = np.linspace(0, n - 1, k).round().astype(int)
    centers = x[order[init_idx]].copy()
    labels = np.zeros(n, dtype=int)
    rng = np.random.default_rng(random_state)
    for iteration in range(max_iter):
        distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if iteration > 0 and np.array_equal(new_labels, labels):
            break
        old_centers = centers.copy()
        labels = new_labels
        for cluster_id in range(k):
            mask = labels == cluster_id
            centers[cluster_id] = x[mask].mean(axis=0) if mask.any() else x[rng.integers(0, n)]
        if np.linalg.norm(centers - old_centers) <= tol:
            break
    inertia = float(((x - centers[labels]) ** 2).sum())
    return labels, centers, inertia


def _pca_projection(matrix):
    x = np.asarray(matrix, dtype=float)
    if x.shape[0] < 2 or x.shape[1] < 2:
        return np.zeros(x.shape[0]), np.zeros(x.shape[0])
    x_centered = x - x.mean(axis=0)
    try:
        u, s, _ = np.linalg.svd(x_centered, full_matrices=False)
        coords = u[:, :2] * s[:2]
        if coords.shape[1] == 1:
            return coords[:, 0], np.zeros(x.shape[0])
        return coords[:, 0], coords[:, 1]
    except Exception:
        return np.zeros(x.shape[0]), np.zeros(x.shape[0])


def _silhouette_score(matrix, labels, max_sample_size=1200, random_state=42):
    x = np.asarray(matrix, dtype=float)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return np.nan
    if len(labels) > max_sample_size:
        rng = np.random.default_rng(random_state)
        sample_indices = []
        per_cluster = max(2, int(np.ceil(max_sample_size / len(unique_labels))))
        for label in unique_labels:
            cluster_indices = np.flatnonzero(labels == label)
            take = min(len(cluster_indices), per_cluster)
            if take:
                sample_indices.extend(rng.choice(cluster_indices, size=take, replace=False).tolist())
        if len(sample_indices) > max_sample_size:
            sample_indices = rng.choice(np.asarray(sample_indices), size=max_sample_size, replace=False).tolist()
        sample_indices = np.asarray(sorted(set(sample_indices)), dtype=int)
        x = x[sample_indices]
        labels = labels[sample_indices]
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
            return np.nan
    distances = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2))
    scores = []
    for idx, label in enumerate(labels):
        same = labels == label
        same[idx] = False
        a = distances[idx, same].mean() if same.any() else 0
        b_values = []
        for other_label in unique_labels:
            if other_label == label:
                continue
            other = labels == other_label
            if other.any():
                b_values.append(distances[idx, other].mean())
        b = min(b_values) if b_values else 0
        denom = max(a, b)
        scores.append((b - a) / denom if denom > 0 else 0)
    return float(np.mean(scores))


def _inertia_from_labels(matrix, labels):
    x = np.asarray(matrix, dtype=float)
    labels = np.asarray(labels)
    if x.size == 0 or len(labels) == 0:
        return np.nan
    inertia = 0.0
    for label in np.unique(labels):
        cluster = x[labels == label]
        if len(cluster) == 0:
            continue
        center = cluster.mean(axis=0)
        inertia += float(((cluster - center) ** 2).sum())
    return inertia


def evaluate_clustering(features_scaled, labels, inertia=None):
    """
    Считает метрики качества кластеризации.
    """
    labels = np.asarray(labels)
    return {
        "silhouette_score": _silhouette_score(features_scaled, labels),
        "inertia": _inertia_from_labels(features_scaled, labels) if inertia is None else inertia,
        "cluster_sizes": pd.Series(labels).value_counts().sort_index().to_dict(),
    }


def _seasonal_columns(features_df):
    share_cols = [c for c in features_df.columns if c.startswith("roomnights_share_m") or c.startswith("gbb_share_m")]
    return share_cols + ["roomnights_SI", "gbb_SI", "roomnights_CV", "gbb_CV", "top3_share", "summer_share", "winter_share", "peak_month"]


def _economic_columns():
    return [
        "log_avg_revenue", "log_avg_roomnights", "log_avg_gbb",
        "ADR", "sales_per_roomnight", "LOS", "is_STR"
    ]


def _complex_columns(features_df):
    return [
        "log_avg_revenue", "log_avg_sales", "log_total_revenue",
        "log_avg_roomnights", "log_avg_gbb", "ADR", "sales_per_roomnight",
        "revenue_per_booking", "LOS", "SI", "CV", "top3_share",
        "summer_share", "winter_share", "growth_revenue_yoy",
        "growth_roomnights_yoy", "growth_gbb_yoy", "median_monthly_yoy",
        "months_available", "active_months", "zero_months", "is_STR"
    ]


def _run_clustering(features_df, k, feature_columns, mode_name):
    features = features_df.copy()
    available = [col for col in feature_columns if col in features.columns]
    usable = []
    for col in available:
        series = pd.to_numeric(features[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if series.notna().sum() >= 3 and series.nunique(dropna=True) > 1:
            usable.append(col)
    if len(usable) < 2 or len(features) < 3:
        result = features.copy()
        result["cluster_id"] = np.nan
        result["cluster_name"] = "Недостаточно данных"
        return result, pd.DataFrame(), {"silhouette_score": np.nan, "inertia": np.nan, "feature_columns": usable}
    scaled, _, _ = _standardize(features, usable)
    labels, _, inertia = _simple_kmeans(scaled.values, k=k, random_state=42)
    result = features.copy()
    result["cluster_id"] = labels + 1
    pc1, pc2 = _pca_projection(scaled.values)
    result["pca_1"] = pc1
    result["pca_2"] = pc2
    result.attrs["feature_columns"] = usable
    result.attrs["features_scaled"] = scaled
    metrics = evaluate_clustering(scaled.values, labels, inertia=inertia)
    metrics["feature_columns"] = usable
    profile = build_cluster_profile(result)
    names = dict(zip(profile["cluster_id"], profile["cluster_name"])) if not profile.empty else {}
    explanations = dict(zip(profile["cluster_id"], profile["cluster_explanation"])) if not profile.empty and "cluster_explanation" in profile.columns else {}
    result["cluster_name"] = result["cluster_id"].map(names).fillna(result["cluster_id"].apply(lambda x: f"??????? {int(x)}"))
    result["cluster_explanation"] = result["cluster_id"].map(explanations).fillna("")
    return result, scaled, metrics


def run_seasonal_clustering(features_df, k):
    """
    Запускает сезонную кластеризацию.
    """
    return _run_clustering(features_df, k, _seasonal_columns(features_df), "seasonal")


def run_economic_clustering(features_df, k):
    """
    Запускает экономическую кластеризацию.
    """
    return _run_clustering(features_df, k, _economic_columns(), "economic")


def run_complex_clustering(features_df, k):
    """
    Запускает итоговую комплексную кластеризацию.
    """
    return _run_clustering(features_df, k, _complex_columns(features_df), "complex")



def _cluster_label(text):
    return text.encode("ascii").decode("unicode_escape")


def _safe_relative_delta(value, baseline):
    if pd.isna(value) or pd.isna(baseline) or abs(float(baseline)) < 1e-9:
        return 0.0
    return float(value - baseline) / max(abs(float(baseline)), 1e-9)


def _build_cluster_traits(row, global_stats):
    medians = global_stats.get("medians", {})
    means = global_stats.get("means", {})
    traits = []

    revenue_delta = _safe_relative_delta(row.get("avg_revenue"), medians.get("avg_monthly_revenue"))
    if revenue_delta >= 0.30:
        traits.append((abs(revenue_delta), _cluster_label("\\u043a\\u0440\\u0443\\u043f\\u043d\\u044b\\u0435"), _cluster_label("\\u0432\\u044b\\u0441\\u043e\\u043a\\u0438\\u0439 \\u0441\\u0440\\u0435\\u0434\\u043d\\u0438\\u0439 revenue")))
    elif revenue_delta <= -0.30:
        traits.append((abs(revenue_delta), _cluster_label("\\u043c\\u0430\\u043b\\u044b\\u0435"), _cluster_label("\\u043d\\u0438\\u0437\\u043a\\u0438\\u0439 \\u0441\\u0440\\u0435\\u0434\\u043d\\u0438\\u0439 revenue")))

    adr_delta = _safe_relative_delta(row.get("ADR"), medians.get("ADR"))
    if adr_delta >= 0.25:
        traits.append((abs(adr_delta), _cluster_label("\\u0434\\u043e\\u0440\\u043e\\u0433\\u0438\\u0435"), _cluster_label("ADR \\u0432\\u044b\\u0448\\u0435 \\u0442\\u0438\\u043f\\u0438\\u0447\\u043d\\u043e\\u0433\\u043e \\u0443\\u0440\\u043e\\u0432\\u043d\\u044f")))
    elif adr_delta <= -0.25:
        traits.append((abs(adr_delta), _cluster_label("\\u0431\\u044e\\u0434\\u0436\\u0435\\u0442\\u043d\\u044b\\u0435"), _cluster_label("ADR \\u043d\\u0438\\u0436\\u0435 \\u0442\\u0438\\u043f\\u0438\\u0447\\u043d\\u043e\\u0433\\u043e \\u0443\\u0440\\u043e\\u0432\\u043d\\u044f")))

    los_delta = _safe_relative_delta(row.get("LOS"), medians.get("LOS"))
    if row.get("str_share", 0) >= 0.60:
        traits.append((1.20, _cluster_label("STR / \\u0430\\u043f\\u0430\\u0440\\u0442\\u0430\\u043c\\u0435\\u043d\\u0442\\u044b"), _cluster_label("\\u0432\\u044b\\u0441\\u043e\\u043a\\u0430\\u044f \\u0434\\u043e\\u043b\\u044f \\u0430\\u043f\\u0430\\u0440\\u0442\\u0430\\u043c\\u0435\\u043d\\u0442\\u043e\\u0432")))
    elif los_delta >= 0.35:
        traits.append((abs(los_delta), _cluster_label("long-stay"), _cluster_label("LOS \\u0432\\u044b\\u0448\\u0435 \\u0442\\u0438\\u043f\\u0438\\u0447\\u043d\\u043e\\u0433\\u043e \\u0443\\u0440\\u043e\\u0432\\u043d\\u044f")))
    elif los_delta <= -0.30:
        traits.append((abs(los_delta), _cluster_label("\\u043a\\u043e\\u0440\\u043e\\u0442\\u043a\\u0438\\u0435 \\u043f\\u0440\\u043e\\u0436\\u0438\\u0432\\u0430\\u043d\\u0438\\u044f"), _cluster_label("LOS \\u043d\\u0438\\u0436\\u0435 \\u0442\\u0438\\u043f\\u0438\\u0447\\u043d\\u043e\\u0433\\u043e \\u0443\\u0440\\u043e\\u0432\\u043d\\u044f")))

    summer_delta = _safe_relative_delta(row.get("summer_share"), means.get("summer_share"))
    winter_delta = _safe_relative_delta(row.get("winter_share"), means.get("winter_share"))
    if row.get("summer_share", 0) >= 0.36 and summer_delta >= 0.15:
        traits.append((abs(summer_delta), _cluster_label("\\u043b\\u0435\\u0442\\u043d\\u0438\\u0439 \\u043f\\u0438\\u043a"), _cluster_label("\\u0432\\u044b\\u0441\\u043e\\u043a\\u0430\\u044f \\u0434\\u043e\\u043b\\u044f \\u043b\\u0435\\u0442\\u043d\\u0435\\u0433\\u043e \\u0441\\u043f\\u0440\\u043e\\u0441\\u0430")))
    if row.get("winter_share", 0) >= 0.32 and winter_delta >= 0.15:
        traits.append((abs(winter_delta), _cluster_label("\\u0437\\u0438\\u043c\\u043d\\u0438\\u0439 \\u043f\\u0438\\u043a"), _cluster_label("\\u0432\\u044b\\u0441\\u043e\\u043a\\u0430\\u044f \\u0434\\u043e\\u043b\\u044f \\u0437\\u0438\\u043c\\u043d\\u0435\\u0433\\u043e \\u0441\\u043f\\u0440\\u043e\\u0441\\u0430")))

    cv_delta = _safe_relative_delta(row.get("CV"), medians.get("CV"))
    if cv_delta <= -0.20:
        traits.append((abs(cv_delta), _cluster_label("\\u0441\\u0442\\u0430\\u0431\\u0438\\u043b\\u044c\\u043d\\u044b\\u0435"), _cluster_label("CV \\u043d\\u0438\\u0436\\u0435 \\u043c\\u0435\\u0434\\u0438\\u0430\\u043d\\u044b, \\u0441\\u043f\\u0440\\u043e\\u0441 \\u0440\\u043e\\u0432\\u043d\\u0435\\u0435")))
    elif cv_delta >= 0.25:
        traits.append((abs(cv_delta), _cluster_label("\\u043d\\u0435\\u0441\\u0442\\u0430\\u0431\\u0438\\u043b\\u044c\\u043d\\u044b\\u0435"), _cluster_label("CV \\u0432\\u044b\\u0448\\u0435 \\u043c\\u0435\\u0434\\u0438\\u0430\\u043d\\u044b, \\u0441\\u043f\\u0440\\u043e\\u0441 \\u043c\\u0435\\u043d\\u0435\\u0435 \\u0440\\u043e\\u0432\\u043d\\u044b\\u0439")))

    growth = row.get("growth_2025_vs_2024")
    if pd.notna(growth):
        if growth >= 0.15:
            traits.append((abs(float(growth)), _cluster_label("\\u0440\\u0430\\u0441\\u0442\\u0443\\u0449\\u0438\\u0435"), _cluster_label("\\u043f\\u043e\\u043b\\u043e\\u0436\\u0438\\u0442\\u0435\\u043b\\u044c\\u043d\\u044b\\u0439 \\u0440\\u043e\\u0441\\u0442 2025 \\u043a 2024")))
        elif growth <= -0.15:
            traits.append((abs(float(growth)), _cluster_label("\\u0441\\u043d\\u0438\\u0436\\u0430\\u044e\\u0449\\u0438\\u0435\\u0441\\u044f"), _cluster_label("\\u043e\\u0442\\u0440\\u0438\\u0446\\u0430\\u0442\\u0435\\u043b\\u044c\\u043d\\u0430\\u044f \\u0434\\u0438\\u043d\\u0430\\u043c\\u0438\\u043a\\u0430 2025 \\u043a 2024")))

    return sorted(traits, key=lambda item: item[0], reverse=True)


def _name_cluster(row, global_stats):
    traits = _build_cluster_traits(row, global_stats)
    labels = []
    for _, label, _ in traits:
        if label not in labels:
            labels.append(label)
        if len(labels) >= 3:
            break
    if not labels:
        labels = [_cluster_label("\\u0441\\u0431\\u0430\\u043b\\u0430\\u043d\\u0441\\u0438\\u0440\\u043e\\u0432\\u0430\\u043d\\u043d\\u044b\\u0435")]
    return " ".join(labels).capitalize()


def _explain_cluster(row, global_stats):
    traits = _build_cluster_traits(row, global_stats)
    reasons = []
    for _, _, reason in traits[:4]:
        if reason not in reasons:
            reasons.append(reason)
    if not reasons:
        reasons = [_cluster_label("\\u043d\\u0435\\u0442 \\u0440\\u0435\\u0437\\u043a\\u043e\\u0433\\u043e \\u043e\\u0442\\u043b\\u0438\\u0447\\u0438\\u044f \\u043e\\u0442 \\u0441\\u0440\\u0435\\u0434\\u043d\\u0435\\u0433\\u043e \\u043f\\u0440\\u043e\\u0444\\u0438\\u043b\\u044f")]
    return "; ".join(reasons)


def build_cluster_profile(clustered_df):
    """
    Формирует таблицу профиля кластеров.
    """
    if clustered_df is None or clustered_df.empty or "cluster_id" not in clustered_df.columns:
        return pd.DataFrame()
    df = clustered_df.dropna(subset=["cluster_id"]).copy()
    if df.empty:
        return pd.DataFrame()
    profile = (
        df.groupby("cluster_id", as_index=False)
        .agg(
            hotels_count=("hotel_id", "count"),
            str_share=("is_STR", "mean"),
            avg_revenue=("avg_monthly_revenue", "mean"),
            median_revenue=("avg_monthly_revenue", "median"),
            avg_roomnights=("avg_monthly_roomnights", "mean"),
            avg_gbb=("avg_monthly_gbb", "mean"),
            ADR=("ADR", "mean"),
            LOS=("LOS", "mean"),
            SI=("SI", "mean"),
            CV=("CV", "mean"),
            summer_share=("summer_share", "mean"),
            winter_share=("winter_share", "mean"),
            growth_2025_vs_2024=("growth_2025_vs_2024", "mean"),
            months_available=("months_available", "mean"),
        )
        .sort_values("cluster_id")
    )
    global_stats = {
        "medians": df[["avg_monthly_revenue", "ADR", "LOS", "CV"]].median(numeric_only=True).to_dict(),
        "means": df[["summer_share", "winter_share"]].mean(numeric_only=True).to_dict(),
    }
    top_months = []
    for cluster_id, group in df.groupby("cluster_id"):
        monthly_cols = [f"roomnights_share_m{m:02d}" for m in range(1, 13)]
        means = group[monthly_cols].mean(numeric_only=True)
        top = means.sort_values(ascending=False).head(3)
        top_months.append({
            "cluster_id": cluster_id,
            "top_demand_months": ", ".join(MONTH_NAMES_SHORT[int(col[-2:])] for col in top.index)
        })
    profile = profile.merge(pd.DataFrame(top_months), on="cluster_id", how="left")
    profile["cluster_name"] = profile.apply(lambda row: _name_cluster(row, global_stats), axis=1)
    profile["cluster_explanation"] = profile.apply(lambda row: _explain_cluster(row, global_stats), axis=1)
    duplicated = profile["cluster_name"].duplicated(keep=False)
    if duplicated.any():
        profile.loc[duplicated, "cluster_name"] = profile.loc[duplicated].apply(
            lambda row: f"{row['cluster_name']} / {row['top_demand_months']}", axis=1
        )
    return profile


def attach_bizdev_actions(clustered_df, actions_df):
    """
    Присоединяет cluster_id к таблице действий BizDev.
    """
    if actions_df is None or actions_df.empty or clustered_df is None or clustered_df.empty:
        return pd.DataFrame()
    mapping = clustered_df[["hotel_id", "cluster_id", "cluster_name"]].drop_duplicates("hotel_id").copy()
    actions = actions_df.copy()
    actions["hotel_id"] = actions["hotel_id"].astype(str).str.strip()
    return actions.merge(mapping, on="hotel_id", how="inner")


def build_bizdev_effect_by_cluster(clustered_df, action_impact):
    if clustered_df is None or clustered_df.empty or action_impact is None or action_impact.empty:
        return pd.DataFrame()
    mapping = clustered_df[["hotel_id", "cluster_id", "cluster_name"]].drop_duplicates("hotel_id").copy()
    actions = action_impact.copy()
    actions["hotel_id"] = actions["hotel_id"].astype(str).str.strip()
    actions = actions.merge(mapping, on="hotel_id", how="inner")
    if actions.empty:
        return pd.DataFrame()
    actions["success_flag"] = actions["manager_effect_pp"] > 0
    grouped_cols = ["cluster_id", "cluster_name", "subject"] if "subject" in actions.columns else ["cluster_id", "cluster_name"]
    summary = (
        actions.groupby(grouped_cols, as_index=False)
        .agg(
            hotels_with_actions=("hotel_id", "nunique"),
            actions_count=("hotel_id", "count"),
            avg_effect=("manager_effect_pp", "mean"),
            median_effect=("manager_effect_pp", "median"),
            success_rate=("success_flag", "mean"),
        )
        .sort_values(["cluster_id", "actions_count"], ascending=[True, False])
    )
    cluster_sizes = clustered_df.groupby("cluster_id", as_index=False).agg(hotels_in_cluster=("hotel_id", "nunique"))
    return summary.merge(cluster_sizes, on="cluster_id", how="left")
