# AGENTS.md

## What this is

A Streamlit dashboard refactored into a 12-file `hotel_dashboard/` package (was monolithic `app_tz_v11_stats_clusters.py` ~4710 lines). Russian UI, English title "Hotel Seasonality & Manager Impact Dashboard". No tests, no CI, no manifests.

## Run it

```bash
pip install streamlit pandas numpy plotly scipy   # scipy is optional
streamlit run hotel_dashboard/app.py
```

Do **not** use plain `python` — the module calls `st.set_page_config`, `st.stop()`, and renders the UI inline at import time, so it only works under `streamlit run`.

There is no `requirements.txt`, `pyproject.toml`, or sample data in the repo.

## Code layout (12 modules)

- `config.py` — constants: `METRICS`, `BOOKING_COLUMNS`, `ACTION_COLUMNS`, `SUBJECT_DESCRIPTIONS`, `GLOSSARY`, `MONTH_NAMES`, `METHODOLOGY_LINKS`, `SCIPY_AVAILABLE` flag.
- `utils.py` — pure helpers: `parse_bool`, `parse_number`, `safe_divide`, `mode_value`, `format_number`, `format_percent`, `classify_outcome_group`, `make_segment_key`, `add_months`, `make_period_column`, `make_period_label`, `normalize_columns`.
- `ui_components.py` — CSS injection, dark Plotly theme, `safe_plotly_chart` wrapper, `glass_card`, `metric_status`, `show_glossary`.
- `data.py` — CSV reading (`read_csv_auto` tries utf-8/cp1251/latin1), `check_columns`, `prepare_bookings_data` / `prepare_actions_data` (both `@st.cache_data`).
- `model.py` — seasonality, expected values, efficiency, individual effect, hotel scores.
- `analysis.py` — sample-quality filter, period classification, action impact, correlation datasets.
- `visualization.py` — all Plotly figure builders (line, bar, heatmap, scatter, t-test).
- `statistics.py` — one-sample t-test, Welch t-test, p-value formatting, `build_ttest_tables`.
- `clustering.py` — feature engineering, hand-written KMeans + PCA, `build_cluster_analysis`.
- `report.py` — `build_main_answer`, `make_report_html` (standalone HTML report).
- `app.py` — **main entry point**, ~1800 lines of inline `st.*` calls across 12 tabs. Edits to the dashboard almost always land here.
- `__init__.py` — empty package marker.

Original monolithic backup: `app_tz_v11_stats_clusters.py` at repo root.

## Hard constraints

- **Required input columns** (app will `st.stop()` if missing):
  - Bookings CSV: `hotel_id, booking_created_date, is_STR, gbb, roomnights, sales_volumes_rub, revenue_rub`.
  - Actions CSV: `action_date, subject, outcome, hotel_id`.
- **`is_STR` parsing** (`parse_bool`): truthy include `true/1/yes/y/да/str/апарт/апарт-отель`; falsy include `false/0/no/n/нет/hotel/отель`. Anything else → `False` (HOTEL).
- **Numeric parsing** (`parse_number`): handles NBSP, spaces, and `,`-as-decimal. Don't replace with `pd.to_numeric` — Russian-locale CSVs will break.
- **`safe_plotly_chart`** wrapper (in `ui_components.py`) — always use for new charts, never `st.plotly_chart` directly (Streamlit 1.5x+ crashes on duplicate Plotly IDs).
- **`@st.cache_data`** on `prepare_bookings_data` / `prepare_actions_data` — clear cache from UI menu or change input shape when modifying logic.
- **Inline UI at module top level** in `app.py` — ~1800 lines of bare `st.*`. Wrap new sections in functions only if wired into the existing tab flow.
- **Russian copy**: all user-facing strings are in Russian. Preserve language and tone.

## Common gotchas

- `st.stop()` is called early if either CSV is missing or `booking_created_date` parses to no months. Need two passing CSVs to develop the rest of the app.
- The app does not persist uploads — every rerun re-reads files via `read_csv_auto` (which `seek(0)`s the buffer).
- `classify_outcome_group` in `utils.py` does a lazy `from config import ...` inside the function body to avoid circular imports.
- The custom KMeans (`simple_kmeans`) and PCA projection are hand-written. Do not introduce `sklearn` without rewriting these.
- `build_main_answer` and `make_report_html` are referenced only in the methodology/answer block, not in the tab UI. Check both before assuming a code path is dead.

## What does not exist here

- No tests, no linter, no formatter config. Don't add pytest/ruff/Black unless the user asks.
- No git repo (`Is directory a git repo: no`). Don't run `git` commands.
- No README, no `opencode.json` in the repo. Keep documentation in this single file unless the user asks for more.
