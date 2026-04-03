"""
Standalone run:  python eda.py   (prints dataset summary)
"""

import os, sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

DATA_PATH = "data/mental_health_dataset.csv"

#Shared style
RISK_COLORS = {
    "Minimal":  "#22c55e",
    "Mild":     "#eab308",
    "Moderate": "#f97316",
    "Severe":   "#ef4444",
}
RISK_ORDER = ["Minimal", "Mild", "Moderate", "Severe"]

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(210,240,232,0.5)",
    font=dict(color="#1a2744", family="Nunito, sans-serif"),
    title_font=dict(size=15, color="#1a5c42"),
    legend=dict(bgcolor="rgba(255,255,255,0.75)", bordercolor="#b8ddd0", borderwidth=1),
    margin=dict(t=50, b=40, l=40, r=20),
    xaxis=dict(gridcolor="#b8ddd0", linecolor="#8ecfbb", tickcolor="#2a6858"),
    yaxis=dict(gridcolor="#b8ddd0", linecolor="#8ecfbb", tickcolor="#2a6858"),
)

#Data loader
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        sys.path.insert(0, "data")
        import generate_dataset  # noqa
    return pd.read_csv(DATA_PATH)

def summary_stats(df: pd.DataFrame) -> dict:
    return {
        "total":         len(df),
        "avg_phq9":      round(df["phq9_score"].mean(), 1),
        "avg_gad7":      round(df["gad7_score"].mean(), 1),
        "avg_stress":    round(df["stress_level"].mean(), 1),
        "avg_sleep":     round(df["sleep_hours"].mean(), 1),
        "high_risk_pct": round(df["risk_level"].isin(["Moderate","Severe"]).mean()*100, 1),
    }

#Chart 1 — Risk distribution bar
def chart_risk_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["risk_level"].value_counts().reindex(RISK_ORDER)
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=[RISK_COLORS[r] for r in counts.index],
        text=counts.values, textposition="outside",
        textfont=dict(color="#1a2744"),
    ))
    fig.update_layout(
        title="Risk Level Distribution",
        xaxis_title="Risk Level", yaxis_title="Count",
        **BASE_LAYOUT,
    )
    return fig

#Chart 2 — Age by risk box
def chart_age_by_risk(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df, x="risk_level", y="age",
        color="risk_level", color_discrete_map=RISK_COLORS,
        category_orders={"risk_level": RISK_ORDER},
        title="Age Distribution by Risk Level",
    )
    fig.update_layout(**BASE_LAYOUT)
    return fig

#Chart 3 — Gender × risk grouped bar
def chart_gender_risk(df: pd.DataFrame) -> go.Figure:
    ct = df.groupby(["gender", "risk_level"]).size().reset_index(name="count")
    fig = px.bar(
        ct, x="gender", y="count", color="risk_level",
        color_discrete_map=RISK_COLORS, barmode="group",
        category_orders={"risk_level": RISK_ORDER},
        title="Risk Level by Gender",
    )
    fig.update_layout(**BASE_LAYOUT)
    return fig

#Chart 4 — Sleep vs PHQ-9 scatter
def chart_sleep_vs_phq(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df, x="sleep_hours", y="phq9_score",
        color="risk_level", color_discrete_map=RISK_COLORS,
        opacity=0.6, trendline="ols",
        title="Sleep Hours vs PHQ-9 Score",
    )
    fig.update_layout(**BASE_LAYOUT)
    return fig

#Chart 5 — Stress vs Exercise scatter
def chart_stress_vs_exercise(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df, x="stress_level", y="exercise_days_per_week",
        color="risk_level", color_discrete_map=RISK_COLORS,
        size="phq9_score", opacity=0.65,
        title="Stress Level vs Exercise Frequency",
    )
    fig.update_layout(**BASE_LAYOUT)
    return fig

#Chart 6 — PHQ-9 vs GAD-7
def chart_phq_vs_gad(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df, x="phq9_score", y="gad7_score",
        color="risk_level", color_discrete_map=RISK_COLORS,
        opacity=0.6, title="PHQ-9 (Depression) vs GAD-7 (Anxiety)",
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=27, y1=21,
                  line=dict(dash="dot", color="#9b7dd4", width=1))
    fig.update_layout(**BASE_LAYOUT)
    return fig

#Chart 7 — Correlation heatmap
def chart_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    cols = [
        "age", "sleep_hours", "stress_level",
        "exercise_days_per_week", "social_support_score",
        "work_hours_per_week", "screen_time_hours",
        "phq9_score", "gad7_score",
    ]
    corr = df[cols].corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu_r", zmid=0,
        text=corr.values, texttemplate="%{text}",
        textfont=dict(size=10, color="#2d1b4e"),
    ))
    fig.update_layout(
        title="Feature Correlation Heatmap", height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(210,240,232,0.5)",
        font=dict(color="#1a2744", family="Nunito, sans-serif"),
        title_font=dict(size=15, color="#1a5c42"),
        legend=dict(bgcolor="rgba(255,255,255,0.75)"),
        margin=dict(t=50, b=40, l=40, r=20),
    )
    return fig

#Chart 8 — Work hours distribution
def chart_work_hours_dist(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="work_hours_per_week", color="risk_level",
        color_discrete_map=RISK_COLORS, nbins=30, opacity=0.8,
        category_orders={"risk_level": RISK_ORDER},
        title="Work Hours per Week Distribution",
        barmode="overlay",
    )
    fig.update_layout(**BASE_LAYOUT)
    return fig
if __name__ == "__main__":
    df = load_data()
    print("Dataset Summary:")
    print(df.describe())
    print("\nRisk distribution:")
    print(df["risk_level"].value_counts())
