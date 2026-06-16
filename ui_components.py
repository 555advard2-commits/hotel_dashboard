import html

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from config import GLOSSARY, METHODOLOGY_LINKS


_PLOTLY_RENDER_COUNTER = 0


def safe_plotly_chart(fig, use_container_width=True, key=None, **kwargs):
    global _PLOTLY_RENDER_COUNTER
    _PLOTLY_RENDER_COUNTER += 1
    base_key = key or "plotly_chart"
    safe_key = f"{base_key}_{_PLOTLY_RENDER_COUNTER}"
    return st.plotly_chart(
        fig,
        use_container_width=use_container_width,
        key=safe_key,
        **kwargs
    )


def inject_global_css():
    st.markdown(
        """
        <style>
        :root {
            --color-accent: #CC7D5E;
            --color-accent-hover: #B96E52;
            --color-bg: #F9F9F7;
            --color-card: #FFFFFF;
            --color-text: #2D2D2B;
            --color-muted: #74716D;
            --color-border: #DDD9D4;
            --color-border-soft: #ECE9E5;
            --color-sidebar: #2D2D2B;
            --color-accent-soft: #F4E5DF;
            --color-success: #4F7A65;
            --color-warning: #C69245;
            --color-error: #CC7D5E;
            --color-info: #59758C;
        }
        html, body, [class*="css"] {
            color: var(--color-text);
            font-family: "Inter", "Segoe UI", Arial, sans-serif;
        }
        .stApp {
            color: var(--color-text);
            background: var(--color-bg);
        }
        .stApp::before {
            content: none;
        }
        [data-testid="stAppViewContainer"] > .main {
            background: var(--color-bg);
        }
        [data-testid="stHeader"] {
            background: var(--color-bg);
        }
        .block-container {
            max-width: 1480px;
            padding: 2rem 2.5rem 4rem;
            position: relative;
            z-index: 1;
        }
        h1, h2, h3, h4, h5, h6,
        p, li, label, span, div[data-testid="stMarkdownContainer"],
        .stMarkdown, .stCaption, [data-testid="stWidgetLabel"] {
            color: var(--color-text);
        }
        .stCaption, caption, .small-muted,
        div[data-testid="stMarkdownContainer"] small,
        .muted-text {
            color: var(--color-muted) !important;
        }
        a {
            color: var(--color-accent) !important;
            text-decoration: none !important;
            font-weight: 750;
        }
        a:hover {
            color: var(--color-accent) !important;
            text-decoration: underline !important;
        }
        [data-testid="stSidebar"] {
            background: var(--color-sidebar) !important;
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        [data-testid="stSidebar"] * {
            color: var(--color-bg) !important;
        }
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] small,
        [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
            color: var(--color-bg) !important;
        }
        div[data-testid="stMetric"] {
            background: var(--color-card);
            border: 1px solid var(--color-border-soft);
            border-radius: 12px;
            padding: 18px 20px;
            box-shadow: 0 2px 8px rgba(45,45,43,0.05);
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: var(--color-muted) !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--color-text) !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
            color: var(--color-accent) !important;
        }
        .corporate-card {
            background: var(--color-card);
            border: 1px solid var(--color-border-soft);
            border-radius: 12px;
            padding: 22px 24px;
            margin: 12px 0 20px;
            box-shadow: 0 2px 8px rgba(45,45,43,0.05);
            color: var(--color-text);
        }
        .corporate-card h3,
        .corporate-card p,
        .corporate-card div,
        .corporate-card li {
            color: var(--color-text);
        }
        .corporate-hero {
            background: var(--color-card);
            border: 1px solid var(--color-border);
            border-left: 5px solid var(--color-accent);
            border-radius: 12px;
            padding: 26px 30px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(45,45,43,0.05);
        }
        .corporate-hero h1 {
            color: var(--color-text);
            font-size: 32px;
            margin: 0 0 8px;
            line-height: 1.13;
            letter-spacing: 0;
        }
        .corporate-hero p {
            color: var(--color-muted);
            margin: 0;
            font-size: 16px;
        }
        .badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 800;
            margin: 3px 5px 3px 0;
            background: var(--color-accent-soft);
            color: var(--color-accent);
            border: 1px solid var(--color-accent);
        }
        .badge-good {
            background: #E4EFE9;
            color: var(--color-success);
            border-color: var(--color-success);
        }
        .badge-warn {
            background: #F6E8D0;
            color: var(--color-warning);
            border-color: var(--color-warning);
        }
        .badge-bad {
            background: var(--color-accent-soft);
            color: var(--color-accent);
            border-color: var(--color-accent);
        }
        .badge-info {
            background: #E1E9EE;
            color: var(--color-info);
            border-color: var(--color-info);
        }
        div[data-testid="stTabs"] button {
            background: transparent !important;
            color: var(--color-muted) !important;
            border: none !important;
            border-bottom: 3px solid transparent !important;
            border-radius: 0;
            margin-right: 4px;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--color-text) !important;
            border-bottom-color: var(--color-accent) !important;
            font-weight: 600;
        }
        div[data-testid="stExpander"] {
            background: var(--color-card);
            border: 1px solid var(--color-border-soft);
            border-radius: 8px;
            overflow: hidden;
        }
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] p,
        div[data-testid="stExpander"] li {
            color: var(--color-text) !important;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div,
        .stTextInput input,
        .stNumberInput input,
        .stDateInput input {
            background: var(--color-card) !important;
            color: var(--color-text) !important;
            border: 1px solid #CFCBC5 !important;
            border-radius: 8px !important;
        }
        [data-testid="stSidebar"] div[data-baseweb="select"] > div,
        [data-testid="stSidebar"] div[data-baseweb="input"] > div,
        [data-testid="stSidebar"] div[data-baseweb="textarea"] > div,
        [data-testid="stSidebar"] .stTextInput input,
        [data-testid="stSidebar"] .stNumberInput input,
        [data-testid="stSidebar"] .stDateInput input {
            background: #393936 !important;
            color: var(--color-bg) !important;
            border-color: rgba(255,255,255,0.14) !important;
        }
        div[data-baseweb="select"] > div:hover,
        div[data-baseweb="input"] > div:hover,
        div[data-baseweb="textarea"] > div:hover {
            border-color: var(--color-accent) !important;
        }
        div[data-baseweb="select"] svg {
            color: currentColor !important;
            fill: currentColor !important;
        }
        [data-baseweb="tag"] {
            background: var(--color-accent) !important;
            color: #FFFFFF !important;
            border: 1px solid var(--color-accent) !important;
            border-radius: 8px !important;
        }
        [data-baseweb="tag"] *,
        [data-baseweb="tag"] svg {
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
        }
        [data-testid="stSidebar"] [data-baseweb="tag"] {
            background: var(--color-accent) !important;
            color: #FFFFFF !important;
            border-color: var(--color-accent) !important;
        }
        input, textarea {
            color: var(--color-text) !important;
        }
        input::placeholder, textarea::placeholder {
            color: var(--color-muted) !important;
        }
        div[data-baseweb="popover"], div[data-baseweb="menu"] {
            background-color: var(--color-card) !important;
            color: var(--color-text) !important;
            border: 1px solid var(--color-border) !important;
        }
        div[role="option"] {
            background-color: var(--color-card) !important;
            color: var(--color-text) !important;
        }
        div[role="option"]:hover {
            background-color: var(--color-accent-soft) !important;
            color: var(--color-text) !important;
        }
        .stSlider [data-baseweb="slider"] div {
            color: var(--color-text) !important;
        }
        .stCheckbox [data-baseweb="checkbox"] div,
        .stCheckbox [data-baseweb="checkbox"] span,
        [data-testid="stCheckbox"] [data-baseweb="checkbox"] div,
        [data-testid="stCheckbox"] [data-baseweb="checkbox"] span {
            border-color: var(--color-accent) !important;
        }
        .stCheckbox [data-baseweb="checkbox"] div[aria-checked="true"],
        [data-testid="stCheckbox"] [data-baseweb="checkbox"] div[aria-checked="true"],
        .stCheckbox [data-baseweb="checkbox"] span[aria-checked="true"],
        [data-testid="stCheckbox"] [data-baseweb="checkbox"] span[aria-checked="true"] {
            background-color: var(--color-accent) !important;
            border-color: var(--color-accent) !important;
        }
        .stCheckbox svg,
        [data-testid="stCheckbox"] svg {
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
            stroke: #FFFFFF !important;
        }
        .stSlider [data-baseweb="slider"] [role="slider"] {
            background-color: var(--color-accent) !important;
            border-color: var(--color-accent) !important;
            box-shadow: none !important;
        }
        .stSlider [data-baseweb="slider"] div[style*="background"] {
            background-color: var(--color-accent) !important;
        }
        .stSlider [data-baseweb="slider"] div[style*="rgb(255, 75, 75)"],
        .stSlider [data-baseweb="slider"] div[style*="#ff4b4b"] {
            background-color: var(--color-accent) !important;
        }
        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"] {
            background: var(--color-card) !important;
            color: var(--color-text) !important;
            border: 1px solid var(--color-border) !important;
            border-radius: 8px !important;
        }
        [data-testid="stFileUploader"] section *,
        [data-testid="stFileUploaderDropzone"] * {
            color: var(--color-text) !important;
        }
        [data-testid="stFileUploader"] button {
            background: var(--color-accent) !important;
            color: #FFFFFF !important;
            border: 1px solid var(--color-accent) !important;
            border-radius: 8px !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploader"] section,
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
            background: #393936 !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            border-radius: 10px !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
        [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] > div,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] div,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] span {
            background: #242423 !important;
            color: var(--color-bg) !important;
            border-color: rgba(255,255,255,0.10) !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
            border: 1px solid rgba(255,255,255,0.10) !important;
            border-radius: 10px !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploader"] section *,
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
            color: var(--color-bg) !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
            background: var(--color-accent) !important;
            color: #FFFFFF !important;
            border: 1px solid var(--color-accent) !important;
        }
        .upload-status {
            background: #393936;
            border: 1px solid rgba(255,255,255,0.14);
            border-left: 4px solid var(--color-accent);
            border-radius: 10px;
            padding: 10px 12px;
            margin: 8px 0 18px;
        }
        .upload-status-title {
            color: var(--color-bg) !important;
            font-weight: 800;
            margin-bottom: 5px;
        }
        .upload-status-body {
            color: var(--color-bg) !important;
            font-size: 13px;
            line-height: 1.35;
            overflow-wrap: anywhere;
        }
        .upload-status-badge {
            display: inline-block;
            background: var(--color-accent);
            color: #FFFFFF !important;
            border-radius: 999px;
            padding: 2px 8px;
            font-size: 12px;
            font-weight: 700;
            margin-bottom: 6px;
        }
        .upload-status-empty {
            border-left-color: var(--color-accent);
        }
        .stButton > button {
            background: var(--color-accent) !important;
            color: #FFFFFF !important;
            border: 1px solid var(--color-accent) !important;
            border-radius: 8px !important;
            font-weight: 600;
            box-shadow: none !important;
        }
        .stButton > button:hover {
            background: var(--color-accent-hover) !important;
            border-color: var(--color-accent-hover) !important;
        }
        .stDownloadButton > button {
            background: var(--color-card) !important;
            color: var(--color-accent) !important;
            border: 1px solid var(--color-accent) !important;
            border-radius: 8px !important;
            font-weight: 600;
            box-shadow: none !important;
        }
        .stDownloadButton > button:hover {
            background: var(--color-accent-soft) !important;
            border-color: var(--color-accent-hover) !important;
            color: var(--color-accent-hover) !important;
        }
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
            background: var(--color-card);
            border: 1px solid var(--color-border-soft);
            border-radius: 10px;
            overflow: hidden;
        }
        div[data-testid="stDataFrame"] * {
            color: var(--color-text);
        }
        [data-testid="stAlert"] {
            background: var(--color-accent-soft) !important;
            color: var(--color-text) !important;
            border: 1px solid var(--color-accent) !important;
            border-radius: 8px;
        }
        [data-testid="stAlert"] * {
            color: var(--color-text) !important;
        }
        pre, code, .stCodeBlock {
            background: var(--color-card) !important;
            color: var(--color-text) !important;
            border-radius: 8px !important;
            border-color: var(--color-border) !important;
        }
        hr {
            border-color: var(--color-border) !important;
        }
        .method-link a {
            text-decoration: none;
            font-weight: 800;
        }
        [data-testid="stDataFrame"] div,
        [data-testid="stDataFrame"] span,
        [data-testid="stDataFrame"] p {
            color: var(--color-text) !important;
        }
        [data-testid="stTooltipContent"],
        [data-baseweb="tooltip"] {
            background: var(--color-sidebar) !important;
            color: var(--color-bg) !important;
            border: 1px solid var(--color-accent) !important;
        }
        [data-testid="stTooltipContent"] *,
        [data-baseweb="tooltip"] * {
            color: var(--color-bg) !important;
        }
        .stApp, .block-container {
            text-rendering: optimizeLegibility;
        }
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li {
            color: var(--color-text) !important;
        }
        div[data-testid="stMarkdownContainer"] strong, b {
            color: var(--color-text) !important;
        }
        div[data-testid="stDataFrame"] {
            background: var(--color-card) !important;
        }
        div[data-testid="stDataFrame"] [role="gridcell"],
        div[data-testid="stDataFrame"] [role="columnheader"] {
            color: var(--color-text) !important;
            background-color: var(--color-card) !important;
        }
        div[data-testid="stDataFrame"] [role="columnheader"] {
            color: var(--color-text) !important;
            font-weight: 800 !important;
        }
        .stTooltipIcon, [data-testid="stTooltipIcon"] {
            color: var(--color-accent) !important;
        }
        [data-testid="stTooltipIcon"] svg,
        [data-testid="stTooltipIcon"] svg *,
        [data-testid="stTooltipIcon"] path,
        button[aria-label*="Help"] svg,
        button[aria-label*="Help"] svg *,
        button[aria-label*="help"] svg,
        button[aria-label*="help"] svg * {
            color: var(--color-accent) !important;
            fill: none !important;
            stroke: var(--color-accent) !important;
        }
        [data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg,
        [data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg *,
        [data-testid="stSidebar"] [data-testid="stTooltipIcon"] path,
        [data-testid="stSidebar"] button[aria-label*="Help"] svg,
        [data-testid="stSidebar"] button[aria-label*="Help"] svg *,
        [data-testid="stSidebar"] button[aria-label*="help"] svg,
        [data-testid="stSidebar"] button[aria-label*="help"] svg * {
            color: var(--color-bg) !important;
            fill: none !important;
            stroke: var(--color-bg) !important;
        }
        [data-testid="stTooltipIcon"] text,
        button[aria-label*="Help"] text,
        button[aria-label*="help"] text {
            fill: currentColor !important;
            stroke: none !important;
        }
        [data-testid="stTooltipContent"],
        [data-testid="stTooltipContent"] *,
        [data-testid="stTooltipContent"] div,
        [data-testid="stTooltipContent"] p,
        [data-testid="stTooltipContent"] span,
        [data-testid="stTooltipContent"] li,
        [data-baseweb="tooltip"],
        [data-baseweb="tooltip"] *,
        [data-baseweb="tooltip"] div,
        [data-baseweb="tooltip"] p,
        [data-baseweb="tooltip"] span,
        [data-baseweb="tooltip"] li {
            color: #F9F9F7 !important;
            -webkit-text-fill-color: #F9F9F7 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def apply_corporate_plotly_theme():
    template = go.layout.Template(
        layout=go.Layout(
            font=dict(
                color="#2D2D2B",
                family="Inter, Segoe UI, Arial, sans-serif"
            ),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            colorway=[
                "#CC7D5E",
                "#59758C",
                "#4F7A65",
                "#C69245",
                "#8B6F91",
                "#6D8078",
            ],
            title=dict(font=dict(color="#2D2D2B", size=20)),
            legend=dict(
                bgcolor="rgba(255,255,255,0)",
                font=dict(color="#74716D")
            ),
            xaxis=dict(
                gridcolor="#ECE9E5",
                zerolinecolor="#DDD9D4",
                linecolor="#DDD9D4",
                tickfont=dict(color="#74716D"),
                title=dict(font=dict(color="#2D2D2B"))
            ),
            yaxis=dict(
                gridcolor="#ECE9E5",
                zerolinecolor="#DDD9D4",
                linecolor="#DDD9D4",
                tickfont=dict(color="#74716D"),
                title=dict(font=dict(color="#2D2D2B"))
            ),
            margin=dict(l=24, r=24, t=72, b=36)
        )
    )
    pio.templates["corporate_light"] = template
    pio.templates.default = "corporate_light"
    px.defaults.template = "corporate_light"
    px.defaults.color_discrete_sequence = [
        "#CC7D5E",
        "#59758C",
        "#4F7A65",
        "#C69245",
        "#8B6F91",
        "#6D8078",
    ]


def corporate_card(title, body, badge_text=None, badge_class="badge"):
    badge_html = f"<span class='{badge_class}'>{html.escape(str(badge_text))}</span>" if badge_text else ""
    st.markdown(
        f"""
        <div class="corporate-card">
            {badge_html}
            <h3 style="margin:8px 0 10px;">{html.escape(str(title))}</h3>
            <div>{body}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def metric_status(efficiency):
    if pd.isna(efficiency):
        return "☐", "недостаточно данных"
    if efficiency >= 1.05:
        return "✅", "выше сезонного ожидания"
    if efficiency <= 0.95:
        return "⚠️", "ниже сезонного ожидания"
    return "☑️", "около сезонного ожидания"


def show_glossary():
    data = pd.DataFrame([{"term": k, "explanation_ru": v} for k, v in GLOSSARY.items()])
    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True,
        column_config={
            "term": st.column_config.TextColumn("Термин", help="Английский термин из данных или методики."),
            "explanation_ru": st.column_config.TextColumn("Пояснение на русском", help="Что термин означает в проекте.")
        }
    )
