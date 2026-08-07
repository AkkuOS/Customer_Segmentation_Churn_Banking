import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="European Banking Churn Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PROJECT PATHS
# =========================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
PROCESSED_DIR = DATA_DIR / "processed"

MODEL_PATH = MODEL_DIR / "final_smote_gradient_boosting.pkl"
SCALER_PATH = MODEL_DIR / "standard_scaler.pkl"
THRESHOLD_PATH = MODEL_DIR / "final_classification_threshold.json"
DATASET_PATH = DATA_DIR / "bank_customer_churn_enriched.csv"
FEATURE_IMPORTANCE_PATH = TABLE_DIR / "feature_importance.csv"
PERMUTATION_IMPORTANCE_PATH = TABLE_DIR / "permutation_importance.csv"
FINAL_PREDICTIONS_PATH = PROCESSED_DIR / "final_test_predictions.csv"


# =========================================================
# POWER BI STYLE PALETTE
# =========================================================

POWERBI_BLUE = "#118DFF"
POWERBI_DARK_BLUE = "#1F4E79"
POWERBI_RED = "#E15759"
POWERBI_ORANGE = "#F28E2B"
POWERBI_GREEN = "#59A14F"
POWERBI_PURPLE = "#7B61FF"
POWERBI_LIGHT_BLUE = "#73C2FB"
POWERBI_GREY = "#64748B"
POWERBI_BG = "#F3F4F6"
POWERBI_CARD = "#FFFFFF"


# =========================================================
# CUSTOM POWER BI STYLE THEME
# =========================================================

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: Inter, Segoe UI, Arial, sans-serif;
    }

    .stApp {
        background-color: #F3F4F6;
    }

    .block-container {
        padding-top: 0.8rem;
        padding-bottom: 2rem;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
        max-width: 100%;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #111827 100%);
        border-right: 1px solid #1F2937;
    }

    [data-testid="stSidebar"] * {
        color: #F8FAFC;
    }

    [data-testid="stHeader"] {
        display: none;
    }

    #MainMenu, footer, header {
        visibility: hidden;
        height: 0;
    }

    .dashboard-title {
        font-size: 2rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0.10rem;
        line-height: 1.15;
    }

    .dashboard-subtitle {
        color:#222222;
        font-size:0.95rem;
        font-weight:500;
        margin-bottom: 1.0rem;
    }

    .page-tag {
        font-size: 0.72rem;
        color: #64748B;
        letter-spacing: 0.08em;
        font-weight: 700;
        text-transform: uppercase;
    }

    .section-title {
        color:#111111;
        font-size:1.05rem;
        font-weight:700;
        margin-top: 0.35rem;
        margin-bottom: 0.55rem;
    }

    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 9px;
        padding: 14px 15px;
        min-height: 105px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
    }

    .kpi-label {
        color: #6B7280;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .kpi-value {
        color: #118DFF;
        font-size: 1.65rem;
        font-weight: 800;
        margin-top: 7px;
        line-height: 1.1;
    }

    .kpi-note {
        color: #94A3B8;
        font-size: 0.70rem;
        margin-top: 6px;
    }

    .insight-card {
        background: #FFFFFF;
        border-left: 4px solid #118DFF;
        border-radius: 7px;
        padding: 12px 14px;
        margin-bottom: 9px;
        color: #1F2937;
        box-shadow: 0 1px 5px rgba(15, 23, 42, 0.04);
    }

    .recommendation-card {
        background: #FFFFFF;
        border-left: 4px solid #59A14F;
        border-radius: 7px;
        padding: 13px 15px;
        margin-bottom: 10px;
        color: #1F2937;
        box-shadow: 0 1px 5px rgba(15, 23, 42, 0.04);
    }

    .risk-low, .risk-medium, .risk-high, .risk-critical {
        padding: 11px 14px;
        border-radius: 8px;
        font-weight: 800;
        text-align: center;
        font-size: 1rem;
    }

    .risk-low { background: #DCFCE7; color: #166534; }
    .risk-medium { background: #FEF3C7; color: #92400E; }
    .risk-high { background: #FFEDD5; color: #9A3412; }
    .risk-critical { background: #FEE2E2; color: #991B1B; }

    [data-testid="stPlotlyChart"] {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 4px 7px 1px 7px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
    }

    .stButton > button,
    .stDownloadButton > button,
    .stFormSubmitButton > button {
        border-radius: 6px;
        font-weight: 700;
    }

    div[data-baseweb="select"] > div {
        border-radius: 6px;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        padding: 10px 12px;
        border-radius: 8px;
    }
    
        /* =======================================================
    INPUT LABELS
    ======================================================= */

    label,
    .stSelectbox label,
    .stNumberInput label,
    .stSlider label,
    .stTextInput label,
    .stTextArea label,
    .stCheckbox label,
    .stRadio label,
    .stMultiSelect label{
        color: #111111 !important;
        font-weight: 700 !important;
        font-size: 15px !important;
    }

    /* Streamlit generated labels */
    [data-testid="stWidgetLabel"] p{
        color:#111111 !important;
        font-weight:700 !important;
        font-size:15px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOADING HELPERS
# =========================================================

@st.cache_resource
def load_model_artifacts():
    missing = [
        str(path)
        for path in (MODEL_PATH, SCALER_PATH, THRESHOLD_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing model artifact(s):\n" + "\n".join(missing)
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        threshold_metadata = json.load(file)

    return model, scaler, threshold_metadata


@st.cache_data
def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    return pd.read_csv(DATASET_PATH)


@st.cache_data
def load_optional_table(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


try:
    model, scaler, threshold_metadata = load_model_artifacts()
    df = load_dataset()
except Exception as exc:
    st.error("The dashboard could not load the required project artifacts.")
    st.exception(exc)
    st.stop()

feature_importance_df = load_optional_table(FEATURE_IMPORTANCE_PATH)
permutation_importance_df = load_optional_table(PERMUTATION_IMPORTANCE_PATH)
final_predictions_df = load_optional_table(FINAL_PREDICTIONS_PATH)

CLASSIFICATION_THRESHOLD = float(
    threshold_metadata.get("classification_threshold", 0.56)
)
MODEL_FEATURE_NAMES = threshold_metadata.get("feature_names", [])

CONTINUOUS_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "EstimatedSalary",
    "BalanceSalaryRatio",
    "ProductsPerYear",
    "AgeTenureRatio",
    "ActiveBalance",
    "CustomerValueScore",
]

AGE_ORDER = ["18-30", "31-40", "41-50", "51-60", "60+"]
TENURE_ORDER = ["New", "Regular", "Loyal", "Long-Term"]
BALANCE_ORDER = [
    "Zero Balance",
    "Low Balance",
    "Medium Balance",
    "High Balance",
    "Premium Balance",
]
ENGAGEMENT_ORDER = [
    "Low Engagement",
    "Moderately Engaged",
    "Highly Engaged",
]


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def kpi_card(label, value, note="", icon="📊"):
    html = (
        '<div class="kpi-card">'
        '<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<div class="kpi-label">{label}</div>'
        f'<div style="font-size:1.15rem;">{icon}</div>'
        '</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-note">{note}</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def page_header(title, subtitle, tag):
    html = (
        '<div style="display:flex;justify-content:space-between;align-items:flex-end;gap:1rem;">'
        '<div>'
        f'<div class="dashboard-title">{title}</div>'
        f'<div class="dashboard-subtitle">{subtitle}</div>'
        '</div>'
        f'<div class="page-tag">{tag}</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def section_title(text):
    st.markdown(
        f'<div class="section-title">{text}</div>',
        unsafe_allow_html=True,
    )


def insight_card(text):
    st.markdown(
        f'<div class="insight-card">{text}</div>',
        unsafe_allow_html=True,
    )


def recommendation_card(title, text):
    html = (
        '<div class="recommendation-card">'
        f'<strong>{title}</strong><br>{text}'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_risk_badge(risk_level):
    class_map = {
        "Low Risk": "risk-low",
        "Medium Risk": "risk-medium",
        "High Risk": "risk-high",
        "Critical Risk": "risk-critical",
    }
    css_class = class_map.get(risk_level, "risk-medium")
    st.markdown(
        f'<div class="{css_class}">{risk_level}</div>',
        unsafe_allow_html=True,
    )


def powerbi_layout(fig, title=None, height=360):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=35, r=20, t=55, b=40),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Arial", size=12, color="#374151"),
        title=dict(
            text=title,
            x=0.02,
            xanchor="left",
            font=dict(size=15, color="#111827"),
        ),
        legend=dict(font=dict(color="#374151")),
        legend_title_text="",
    )
    fig.update_xaxes(
        gridcolor="#EEF2F7",
        zeroline=False,
        tickfont=dict(color="#4B5563"),
        title_font=dict(color="#4B5563"),
    )
    fig.update_yaxes(
        gridcolor="#EEF2F7",
        zeroline=False,
        tickfont=dict(color="#4B5563"),
        title_font=dict(color="#4B5563"),
    )
    return fig


def churn_rate_table(data, group_col):
    result = (
        data.groupby(group_col, observed=False)
        .agg(Customers=("CustomerId", "count"), Churn_Rate=("Exited", "mean"))
        .reset_index()
    )
    result["Churn_Rate"] = (result["Churn_Rate"] * 100).round(2)
    return result


def classify_risk_level(probability):
    if probability < 0.30:
        return "Low Risk"
    if probability < CLASSIFICATION_THRESHOLD:
        return "Medium Risk"
    if probability < 0.75:
        return "High Risk"
    return "Critical Risk"


def prepare_customer_features(
    credit_score,
    geography,
    gender,
    age,
    tenure,
    balance,
    number_of_products,
    has_credit_card,
    is_active_member,
    estimated_salary,
):
    customer_features = pd.DataFrame(
        {
            "CreditScore": [float(credit_score)],
            "Age": [float(age)],
            "Tenure": [float(tenure)],
            "Balance": [float(balance)],
            "NumOfProducts": [int(number_of_products)],
            "HasCrCard": [int(has_credit_card)],
            "IsActiveMember": [int(is_active_member)],
            "EstimatedSalary": [float(estimated_salary)],
            "Geography_Germany": [int(geography == "Germany")],
            "Geography_Spain": [int(geography == "Spain")],
            "Gender_Male": [int(gender == "Male")],
        }
    )

    customer_features["BalanceSalaryRatio"] = (
        customer_features["Balance"]
        / (customer_features["EstimatedSalary"] + 1)
    ).round(4)

    customer_features["ProductsPerYear"] = (
        customer_features["NumOfProducts"]
        / (customer_features["Tenure"] + 1)
    ).round(4)

    customer_features["AgeTenureRatio"] = (
        customer_features["Age"]
        / (customer_features["Tenure"] + 1)
    ).round(4)

    customer_features["ActiveBalance"] = (
        customer_features["Balance"]
        * customer_features["IsActiveMember"]
    )

    customer_features["ProductEngagementScore"] = (
        customer_features["NumOfProducts"]
        * (customer_features["IsActiveMember"] + 1)
    )

    customer_features["CustomerValueScore"] = (
        (customer_features["Balance"] / 1000)
        + (customer_features["NumOfProducts"] * 10)
        + (customer_features["IsActiveMember"] * 20)
    ).round(2)

    if not MODEL_FEATURE_NAMES:
        raise ValueError("Feature names are missing from threshold metadata.")

    customer_features = customer_features[MODEL_FEATURE_NAMES].copy()
    customer_features[CONTINUOUS_FEATURES] = scaler.transform(
        customer_features[CONTINUOUS_FEATURES]
    )
    return customer_features


def predict_customer_churn(**customer_inputs):
    prepared_features = prepare_customer_features(**customer_inputs)
    probability = float(model.predict_proba(prepared_features)[0, 1])
    prediction = int(probability >= CLASSIFICATION_THRESHOLD)
    return {
        "probability": probability,
        "prediction": prediction,
        "risk_level": classify_risk_level(probability),
        "prepared_features": prepared_features,
    }


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    (
        '<div style="padding-top:0.35rem;padding-bottom:1.1rem;">'
        '<div style="font-size:1.4rem;font-weight:800;letter-spacing:0.02em;">'
        '🏦 BANKWISE'
        '</div>'
        '<div style="font-size:0.9rem;color:#94A3B8;line-height:1.45;margin-top:4px;">'
        'European Banking<br>Customer Analytics<br>by Abhijith Os'
        '</div>'
        '</div>'
    ),
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Dashboard",
    [
        "Executive Overview",
        "Customer Segmentation",
        "Churn Analytics",
        "High-Value Customers",
        "Customer Prediction",
        "Model Insights",
        "Recommendations",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("SMOTE Gradient Boosting")
st.sidebar.caption(f"Prediction threshold: {CLASSIFICATION_THRESHOLD:.2f}")


# =========================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# =========================================================

def render_executive_overview():
    page_header(
        "Executive Overview",
        "European Banking Customer Segmentation & Churn Pattern Analytics",
        "Executive Intelligence",
    )

    total_customers = len(df)
    churned_customers = int(df["Exited"].sum())
    churn_rate = df["Exited"].mean() * 100
    retention_rate = 100 - churn_rate
    active_rate = df["IsActiveMember"].mean() * 100
    average_balance = df["Balance"].mean()

    cols = st.columns(5)
    cards = [
        ("Total Customers", f"{total_customers:,}", "European customer base", "👥"),
        ("Churn Rate", f"{churn_rate:.2f}%", f"{churned_customers:,} exited", "📉"),
        ("Retention Rate", f"{retention_rate:.2f}%", "Customers retained", "🛡️"),
        ("Active Members", f"{active_rate:.2f}%", "Engaged customers", "⚡"),
        ("Avg. Balance", f"{average_balance:,.0f}", "Across all customers", "💰"),
    ]
    for col, card in zip(cols, cards):
        with col:
            kpi_card(*card)

    section_title("Customer & Churn Overview")

    c1, c2 = st.columns([1.15, 1])

    country_summary = df["Geography"].value_counts().reset_index()
    country_summary.columns = ["Country", "Customers"]
    fig_country = px.bar(
        country_summary,
        x="Country",
        y="Customers",
        text="Customers",
        color_discrete_sequence=[POWERBI_BLUE],
    )
    fig_country.update_traces(textposition="outside")
    fig_country = powerbi_layout(fig_country, "Customer Distribution by Country")
    with c1:
        st.plotly_chart(fig_country, use_container_width=True)

    churn_summary = df["Churn_Status"].value_counts().reset_index()
    churn_summary.columns = ["Status", "Customers"]
    fig_churn = px.pie(
        churn_summary,
        names="Status",
        values="Customers",
        hole=0.62,
        color="Status",
        color_discrete_map={"Retained": POWERBI_BLUE, "Churned": POWERBI_RED},
    )
    fig_churn = powerbi_layout(fig_churn, "Retention vs Churn")
    with c2:
        st.plotly_chart(fig_churn, use_container_width=True)

    c3, c4 = st.columns(2)

    geo_churn = churn_rate_table(df, "Geography")
    fig_geo = px.bar(
        geo_churn,
        x="Geography",
        y="Churn_Rate",
        text="Churn_Rate",
        color="Churn_Rate",
        color_continuous_scale=[POWERBI_LIGHT_BLUE, POWERBI_ORANGE, POWERBI_RED],
    )
    fig_geo.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_geo.update_coloraxes(showscale=False)
    fig_geo = powerbi_layout(fig_geo, "Churn Rate by Country")
    fig_geo.update_yaxes(title="Churn Rate (%)")
    with c3:
        st.plotly_chart(fig_geo, use_container_width=True)

    age_churn = churn_rate_table(df, "AgeGroup")
    age_churn["AgeGroup"] = pd.Categorical(
        age_churn["AgeGroup"], categories=AGE_ORDER, ordered=True
    )
    age_churn = age_churn.sort_values("AgeGroup")
    fig_age = px.line(
        age_churn,
        x="AgeGroup",
        y="Churn_Rate",
        markers=True,
        color_discrete_sequence=[POWERBI_BLUE],
    )
    fig_age = powerbi_layout(fig_age, "Churn Rate by Age Group")
    fig_age.update_yaxes(title="Churn Rate (%)")
    with c4:
        st.plotly_chart(fig_age, use_container_width=True)

    section_title("Executive Insights")
    for text in [
        "Germany is the highest-risk market, with a churn rate of 32.44%.",
        "Customers aged 51–60 show the highest age-group churn rate at 56.21%.",
        "Inactive customers are approximately 1.88× more likely to churn than active customers.",
        "Customers holding exactly two products have the strongest observed retention.",
        "High-value customer churn is associated with approximately 58.71 million in balances.",
    ]:
        insight_card(text)


# =========================================================
# PAGE 2 — CUSTOMER SEGMENTATION
# =========================================================

def render_customer_segmentation():
    page_header(
        "Customer Segmentation",
        "Demographic, financial and behavioural segmentation of the European customer base",
        "Customer Intelligence",
    )

    section_title("Filters")
    s1, s2, s3, s4, s5 = st.columns(5)

    with s1:
        selected_country = st.selectbox(
            "Country", ["All Countries", "France", "Germany", "Spain"]
        )
    with s2:
        selected_gender = st.selectbox(
            "Gender", ["All Genders", "Female", "Male"]
        )
    with s3:
        selected_age_group = st.selectbox(
            "Age Group", ["All Age Groups"] + AGE_ORDER
        )
    with s4:
        selected_balance_segment = st.selectbox(
            "Balance Segment", ["All Balance Segments"] + BALANCE_ORDER
        )
    with s5:
        selected_engagement_segment = st.selectbox(
            "Engagement", ["All Engagement Levels"] + ENGAGEMENT_ORDER
        )

    filtered_df = df.copy()
    if selected_country != "All Countries":
        filtered_df = filtered_df[filtered_df["Geography"] == selected_country]
    if selected_gender != "All Genders":
        filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]
    if selected_age_group != "All Age Groups":
        filtered_df = filtered_df[filtered_df["AgeGroup"] == selected_age_group]
    if selected_balance_segment != "All Balance Segments":
        filtered_df = filtered_df[
            filtered_df["BalanceSegment"] == selected_balance_segment
        ]
    if selected_engagement_segment != "All Engagement Levels":
        filtered_df = filtered_df[
            filtered_df["EngagementSegment"] == selected_engagement_segment
        ]

    if filtered_df.empty:
        st.warning("No customers match the selected filters.")
        return

    filtered_customers = len(filtered_df)
    filtered_churn_rate = filtered_df["Exited"].mean() * 100
    filtered_active_rate = filtered_df["IsActiveMember"].mean() * 100
    filtered_average_balance = filtered_df["Balance"].mean()
    filtered_high_value = int(
        (filtered_df["HighValueCustomer"] == "High Value").sum()
    )

    cols = st.columns(5)
    cards = [
        ("Filtered Customers", f"{filtered_customers:,}", "Current segment", "👥"),
        ("Churn Rate", f"{filtered_churn_rate:.2f}%", "Filtered segment", "📉"),
        ("Active Rate", f"{filtered_active_rate:.2f}%", "Customer engagement", "⚡"),
        ("Avg. Balance", f"{filtered_average_balance:,.0f}", "Filtered customers", "💰"),
        ("High-Value Customers", f"{filtered_high_value:,}", "Premium segment", "⭐"),
    ]
    for col, card in zip(cols, cards):
        with col:
            kpi_card(*card)

    section_title("Segment Profile")
    c1, c2 = st.columns(2)

    age_segment = (
        filtered_df["AgeGroup"]
        .value_counts()
        .reindex(AGE_ORDER)
        .fillna(0)
        .reset_index()
    )
    age_segment.columns = ["Age Group", "Customers"]
    fig_age = px.bar(
        age_segment,
        x="Age Group",
        y="Customers",
        text="Customers",
        color_discrete_sequence=[POWERBI_BLUE],
    )
    fig_age.update_traces(textposition="outside")
    fig_age = powerbi_layout(fig_age, "Customer Distribution by Age Group")
    with c1:
        st.plotly_chart(fig_age, use_container_width=True)

    balance_segment = (
        filtered_df["BalanceSegment"]
        .value_counts()
        .reindex(BALANCE_ORDER)
        .fillna(0)
        .reset_index()
    )
    balance_segment.columns = ["Balance Segment", "Customers"]
    fig_balance = px.bar(
        balance_segment,
        x="Balance Segment",
        y="Customers",
        text="Customers",
        color_discrete_sequence=[POWERBI_BLUE],
    )
    fig_balance.update_traces(textposition="outside")
    fig_balance.update_xaxes(tickangle=15)
    fig_balance = powerbi_layout(fig_balance, "Customer Distribution by Balance Segment")
    with c2:
        st.plotly_chart(fig_balance, use_container_width=True)

    c3, c4 = st.columns(2)

    engagement_segment = (
        filtered_df["EngagementSegment"]
        .value_counts()
        .reindex(ENGAGEMENT_ORDER)
        .fillna(0)
        .reset_index()
    )
    engagement_segment.columns = ["Engagement Segment", "Customers"]
    fig_engagement = px.pie(
        engagement_segment,
        names="Engagement Segment",
        values="Customers",
        hole=0.58,
        color="Engagement Segment",
        color_discrete_map={
            "Low Engagement": POWERBI_RED,
            "Moderately Engaged": POWERBI_BLUE,
            "Highly Engaged": POWERBI_GREEN,
        },
    )
    fig_engagement = powerbi_layout(fig_engagement, "Customer Engagement Mix")
    with c3:
        st.plotly_chart(fig_engagement, use_container_width=True)

    tenure_segment = (
        filtered_df["TenureGroup"]
        .value_counts()
        .reindex(TENURE_ORDER)
        .fillna(0)
        .reset_index()
    )
    tenure_segment.columns = ["Tenure Group", "Customers"]
    fig_tenure = px.bar(
        tenure_segment,
        x="Tenure Group",
        y="Customers",
        text="Customers",
        color_discrete_sequence=[POWERBI_BLUE],
    )
    fig_tenure.update_traces(textposition="outside")
    fig_tenure = powerbi_layout(fig_tenure, "Customer Distribution by Tenure")
    with c4:
        st.plotly_chart(fig_tenure, use_container_width=True)

    section_title("Segment Risk Matrix")
    churn_matrix_data = (
        filtered_df.groupby(["AgeGroup", "EngagementSegment"], observed=False)
        .agg(
            Customers=("CustomerId", "count"),
            Churn_Rate=("Exited", "mean"),
        )
        .reset_index()
    )
    churn_matrix_data["Churn_Rate"] *= 100
    churn_matrix_pivot = (
        churn_matrix_data.pivot(
            index="EngagementSegment",
            columns="AgeGroup",
            values="Churn_Rate",
        )
        .reindex(index=ENGAGEMENT_ORDER, columns=AGE_ORDER)
    )
    fig_matrix = px.imshow(
        churn_matrix_pivot,
        text_auto=".1f",
        aspect="auto",
        labels=dict(
            x="Age Group",
            y="Engagement Segment",
            color="Churn Rate (%)",
        ),
        color_continuous_scale=[
            "#E8F3FF",
            POWERBI_LIGHT_BLUE,
            POWERBI_ORANGE,
            POWERBI_RED,
        ],
    )
    fig_matrix = powerbi_layout(
        fig_matrix,
        "Churn Rate by Age Group and Engagement Segment",
        height=350,
    )
    st.plotly_chart(fig_matrix, use_container_width=True)

    with st.expander("View Filtered Customer Data"):
        display_columns = [
            "CustomerId",
            "Geography",
            "Gender",
            "Age",
            "AgeGroup",
            "Balance",
            "BalanceSegment",
            "Tenure",
            "TenureGroup",
            "NumOfProducts",
            "Activity_Status",
            "EngagementSegment",
            "Churn_Status",
        ]
        display_columns = [c for c in display_columns if c in filtered_df.columns]
        st.dataframe(
            filtered_df[display_columns],
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download Filtered Segment Data",
            data=filtered_df.to_csv(index=False),
            file_name="bank_customer_segment.csv",
            mime="text/csv",
        )


# =========================================================
# PAGE 3 — CHURN ANALYTICS
# =========================================================

def render_churn_analytics():
    page_header(
        "Churn Analytics",
        "Investigate the demographic, financial and behavioural patterns associated with customer exits",
        "Risk Analytics",
    )

    churned = int(df["Exited"].sum())
    churn_rate = df["Exited"].mean() * 100
    retention_rate = 100 - churn_rate
    active_churn = df.loc[df["IsActiveMember"] == 1, "Exited"].mean() * 100
    inactive_churn = df.loc[df["IsActiveMember"] == 0, "Exited"].mean() * 100

    cols = st.columns(5)
    cards = [
        ("Overall Churn", f"{churn_rate:.2f}%", "Customer exit rate", "📉"),
        ("Churned Customers", f"{churned:,}", "Exited customers", "🚪"),
        ("Retention Rate", f"{retention_rate:.2f}%", "Customers retained", "🛡️"),
        ("Active Churn", f"{active_churn:.2f}%", "Active members", "⚡"),
        ("Inactive Churn", f"{inactive_churn:.2f}%", "Inactive members", "⚠️"),
    ]
    for col, card in zip(cols, cards):
        with col:
            kpi_card(*card)

    section_title("Demographic Churn Patterns")
    c1, c2, c3 = st.columns(3)

    geo = churn_rate_table(df, "Geography")
    fig_geo = px.bar(
        geo,
        x="Geography",
        y="Churn_Rate",
        text="Churn_Rate",
        color="Churn_Rate",
        color_continuous_scale=[POWERBI_LIGHT_BLUE, POWERBI_ORANGE, POWERBI_RED],
    )
    fig_geo.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_geo.update_coloraxes(showscale=False)
    fig_geo = powerbi_layout(fig_geo, "Churn by Country", 330)
    with c1:
        st.plotly_chart(fig_geo, use_container_width=True)

    gender = churn_rate_table(df, "Gender")
    fig_gender = px.bar(
        gender,
        x="Gender",
        y="Churn_Rate",
        text="Churn_Rate",
        color="Gender",
        color_discrete_map={"Female": POWERBI_PURPLE, "Male": POWERBI_BLUE},
    )
    fig_gender.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_gender = powerbi_layout(fig_gender, "Churn by Gender", 330)
    with c2:
        st.plotly_chart(fig_gender, use_container_width=True)

    age = churn_rate_table(df, "AgeGroup")
    age["AgeGroup"] = pd.Categorical(age["AgeGroup"], categories=AGE_ORDER, ordered=True)
    age = age.sort_values("AgeGroup")
    fig_age = px.bar(
        age,
        x="AgeGroup",
        y="Churn_Rate",
        text="Churn_Rate",
        color="Churn_Rate",
        color_continuous_scale=[POWERBI_LIGHT_BLUE, POWERBI_ORANGE, POWERBI_RED],
    )
    fig_age.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_age.update_coloraxes(showscale=False)
    fig_age = powerbi_layout(fig_age, "Churn by Age Group", 330)
    with c3:
        st.plotly_chart(fig_age, use_container_width=True)

    section_title("Behavioural & Product Churn Patterns")
    c4, c5, c6 = st.columns(3)

    product = churn_rate_table(df, "NumOfProducts")
    fig_product = px.bar(
        product,
        x="NumOfProducts",
        y="Churn_Rate",
        text="Churn_Rate",
        color_discrete_sequence=[POWERBI_BLUE],
    )
    fig_product.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_product = powerbi_layout(fig_product, "Churn by Number of Products", 330)
    with c4:
        st.plotly_chart(fig_product, use_container_width=True)

    activity = churn_rate_table(df, "Activity_Status")
    fig_activity = px.bar(
        activity,
        x="Activity_Status",
        y="Churn_Rate",
        text="Churn_Rate",
        color="Activity_Status",
        color_discrete_map={"Active": POWERBI_GREEN, "Inactive": POWERBI_RED},
    )
    fig_activity.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_activity = powerbi_layout(fig_activity, "Churn by Activity Status", 330)
    with c5:
        st.plotly_chart(fig_activity, use_container_width=True)

    card = churn_rate_table(df, "Credit_Card_Status")
    fig_card = px.bar(
        card,
        x="Credit_Card_Status",
        y="Churn_Rate",
        text="Churn_Rate",
        color_discrete_sequence=[POWERBI_DARK_BLUE],
    )
    fig_card.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_card = powerbi_layout(fig_card, "Churn by Credit Card Ownership", 330)
    with c6:
        st.plotly_chart(fig_card, use_container_width=True)

    section_title("Financial & Relationship Churn Patterns")
    c7, c8 = st.columns(2)

    balance = churn_rate_table(df, "BalanceSegment")
    balance["BalanceSegment"] = pd.Categorical(
        balance["BalanceSegment"], categories=BALANCE_ORDER, ordered=True
    )
    balance = balance.sort_values("BalanceSegment")
    fig_balance = px.line(
        balance,
        x="BalanceSegment",
        y="Churn_Rate",
        markers=True,
        color_discrete_sequence=[POWERBI_ORANGE],
    )
    fig_balance.update_xaxes(tickangle=15)
    fig_balance = powerbi_layout(fig_balance, "Churn by Balance Segment")
    with c7:
        st.plotly_chart(fig_balance, use_container_width=True)

    tenure = churn_rate_table(df, "TenureGroup")
    tenure["TenureGroup"] = pd.Categorical(
        tenure["TenureGroup"], categories=TENURE_ORDER, ordered=True
    )
    tenure = tenure.sort_values("TenureGroup")
    fig_tenure = px.bar(
        tenure,
        x="TenureGroup",
        y="Churn_Rate",
        text="Churn_Rate",
        color_discrete_sequence=[POWERBI_BLUE],
    )
    fig_tenure.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_tenure = powerbi_layout(fig_tenure, "Churn by Tenure Group")
    with c8:
        st.plotly_chart(fig_tenure, use_container_width=True)

    section_title("Top Observed Risk Segments")
    risk_table = (
        df.groupby(["Geography", "Gender", "AgeGroup"], observed=False)
        .agg(Customers=("CustomerId", "count"), Churn_Rate=("Exited", "mean"))
        .reset_index()
    )
    risk_table["Churn_Rate"] = (risk_table["Churn_Rate"] * 100).round(2)
    risk_table = risk_table[risk_table["Customers"] >= 25]
    risk_table = risk_table.sort_values("Churn_Rate", ascending=False).head(12)
    st.dataframe(risk_table, use_container_width=True, hide_index=True)


# =========================================================
# PAGE 4 — HIGH-VALUE CUSTOMERS
# =========================================================

def render_high_value_customers():
    page_header(
        "High-Value Customers",
        "Financial exposure, premium customer churn and retention priorities",
        "Value Protection",
    )

    hv = df[df["HighValueCustomer"] == "High Value"].copy()
    hv_churned = hv[hv["Exited"] == 1].copy()

    threshold = hv["Balance"].min() if not hv.empty else np.nan
    hv_churn_rate = hv["Exited"].mean() * 100 if not hv.empty else 0
    balance_at_risk = hv_churned["Balance"].sum()
    avg_hv_balance = hv["Balance"].mean() if not hv.empty else 0

    cols = st.columns(5)
    cards = [
        ("High-Value Customers", f"{len(hv):,}", "Premium customer base", "⭐"),
        ("High-Value Churn", f"{hv_churn_rate:.2f}%", "Premium churn rate", "📉"),
        ("Churned High-Value", f"{len(hv_churned):,}", "Exited premium customers", "🚪"),
        ("Balance at Risk", f"{balance_at_risk/1_000_000:.2f}M", "Churned high-value balances", "💸"),
        ("Avg. HV Balance", f"{avg_hv_balance:,.0f}", "Average premium balance", "💰"),
    ]
    for col, card in zip(cols, cards):
        with col:
            kpi_card(*card)

    section_title("High-Value Risk Profile")
    c1, c2 = st.columns(2)

    geo = churn_rate_table(hv, "Geography")
    fig_geo = px.bar(
        geo,
        x="Geography",
        y="Churn_Rate",
        text="Churn_Rate",
        color="Churn_Rate",
        color_continuous_scale=[POWERBI_LIGHT_BLUE, POWERBI_ORANGE, POWERBI_RED],
    )
    fig_geo.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_geo.update_coloraxes(showscale=False)
    fig_geo = powerbi_layout(fig_geo, "High-Value Churn by Country")
    with c1:
        st.plotly_chart(fig_geo, use_container_width=True)

    engagement = churn_rate_table(hv, "EngagementSegment")
    engagement["EngagementSegment"] = pd.Categorical(
        engagement["EngagementSegment"], categories=ENGAGEMENT_ORDER, ordered=True
    )
    engagement = engagement.sort_values("EngagementSegment")
    fig_eng = px.bar(
        engagement,
        x="EngagementSegment",
        y="Churn_Rate",
        text="Churn_Rate",
        color="EngagementSegment",
        color_discrete_map={
            "Low Engagement": POWERBI_RED,
            "Moderately Engaged": POWERBI_BLUE,
            "Highly Engaged": POWERBI_GREEN,
        },
    )
    fig_eng.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_eng = powerbi_layout(fig_eng, "High-Value Churn by Engagement")
    with c2:
        st.plotly_chart(fig_eng, use_container_width=True)

    c3, c4 = st.columns(2)

    age = churn_rate_table(hv, "AgeGroup")
    age["AgeGroup"] = pd.Categorical(age["AgeGroup"], categories=AGE_ORDER, ordered=True)
    age = age.sort_values("AgeGroup")
    fig_age = px.line(
        age,
        x="AgeGroup",
        y="Churn_Rate",
        markers=True,
        color_discrete_sequence=[POWERBI_RED],
    )
    fig_age = powerbi_layout(fig_age, "High-Value Churn by Age Group")
    with c3:
        st.plotly_chart(fig_age, use_container_width=True)

    product = churn_rate_table(hv, "NumOfProducts")
    fig_product = px.bar(
        product,
        x="NumOfProducts",
        y="Churn_Rate",
        text="Churn_Rate",
        color_discrete_sequence=[POWERBI_BLUE],
    )
    fig_product.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_product = powerbi_layout(fig_product, "High-Value Churn by Products")
    with c4:
        st.plotly_chart(fig_product, use_container_width=True)

    section_title("High-Value Customer Explorer")
    st.markdown(
    """
    <div style="
        color:#111111;
        font-size:15px;
        font-weight:500;
        margin-bottom:16px;
    ">
        Use this table to identify premium customers and understand
        the financial profile of churned high-value accounts.
    </div>
    """,
    unsafe_allow_html=True
    )
    display_cols = [
        "CustomerId",
        "Geography",
        "Gender",
        "Age",
        "Balance",
        "NumOfProducts",
        "Activity_Status",
        "EngagementSegment",
        "Churn_Status",
    ]
    display_cols = [c for c in display_cols if c in hv.columns]
    st.dataframe(
        hv.sort_values("Balance", ascending=False)[display_cols],
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download High-Value Customer Data",
        data=hv.to_csv(index=False),
        file_name="high_value_customers.csv",
        mime="text/csv",
    )


# =========================================================
# PAGE 5 — CUSTOMER PREDICTION
# =========================================================

def render_customer_prediction():
    page_header(
        "Customer Prediction",
        "Estimate individual churn probability and assign an actionable risk category",
        "Predictive Analytics",
    )

    with st.form("customer_prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            credit_score = st.number_input("Credit Score", 300, 900, 650)
            geography = st.selectbox("Country", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.number_input("Age", 18, 100, 40)

        with c2:
            tenure = st.number_input("Tenure (Years)", 0, 10, 5)
            balance = st.number_input(
                "Account Balance", min_value=0.0, value=75000.0, step=1000.0
            )
            number_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])

        with c3:
            has_credit_card_label = st.selectbox("Credit Card Ownership", ["Yes", "No"])
            is_active_label = st.selectbox("Active Member", ["Yes", "No"])
            estimated_salary = st.number_input(
                "Estimated Salary", min_value=0.0, value=100000.0, step=1000.0
            )

        submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

    if not submitted:
        insight_card(
            "The model uses the same 17-feature preprocessing pipeline validated in Notebook 07. Enter customer details above to generate a churn probability and risk level."
        )
        return

    try:
        result = predict_customer_churn(
            credit_score=credit_score,
            geography=geography,
            gender=gender,
            age=age,
            tenure=tenure,
            balance=balance,
            number_of_products=number_of_products,
            has_credit_card=1 if has_credit_card_label == "Yes" else 0,
            is_active_member=1 if is_active_label == "Yes" else 0,
            estimated_salary=estimated_salary,
        )
    except Exception as exc:
        st.error("Prediction failed. Please verify the saved model, scaler and feature metadata.")
        st.exception(exc)
        return

    probability = result["probability"]
    risk_level = result["risk_level"]
    prediction_label = "Likely to Churn" if result["prediction"] else "Likely to Stay"

    section_title("Prediction Result")
    cols = st.columns(4)
    cards = [
        ("Churn Probability", f"{probability:.2%}", "Model probability", "🎯"),
        ("Risk Level", risk_level, "Probability band", "🚦"),
        ("Prediction", prediction_label, "Threshold decision", "🤖"),
        ("Decision Threshold", f"{CLASSIFICATION_THRESHOLD:.0%}", "Validated threshold", "⚙️"),
    ]
    for col, card in zip(cols, cards):
        with col:
            kpi_card(*card)

    c1, c2 = st.columns([1.2, 1])

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 38}},
            title={"text": "Predicted Churn Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": POWERBI_BLUE},
                "steps": [
                    {"range": [0, 30], "color": "#DCFCE7"},
                    {"range": [30, CLASSIFICATION_THRESHOLD * 100], "color": "#FEF3C7"},
                    {"range": [CLASSIFICATION_THRESHOLD * 100, 75], "color": "#FFEDD5"},
                    {"range": [75, 100], "color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": POWERBI_RED, "width": 4},
                    "thickness": 0.8,
                    "value": CLASSIFICATION_THRESHOLD * 100,
                },
            },
        )
    )
    gauge.update_layout(height=360, margin=dict(l=30, r=30, t=60, b=20))
    with c1:
        st.plotly_chart(gauge, use_container_width=True)

    with c2:
        section_title("Risk Classification")
        render_risk_badge(risk_level)
        st.write("")

        if risk_level == "Low Risk":
            action = "Maintain regular service quality and standard relationship engagement."
        elif risk_level == "Medium Risk":
            action = "Monitor engagement and consider targeted cross-selling or loyalty communication."
        elif risk_level == "High Risk":
            action = "Launch a proactive retention campaign and review product fit and engagement."
        else:
            action = "Prioritize immediate relationship-manager intervention and a personalized retention offer."

        recommendation_card("Suggested Retention Action", action)
        insight_card(
            f"The classification threshold is {CLASSIFICATION_THRESHOLD:.2f}; customers at or above this probability are predicted to churn."
        )

    report = pd.DataFrame(
        {
            "Metric": [
                "Prediction",
                "Churn Probability",
                "Risk Level",
                "Classification Threshold",
            ],
            "Value": [
                prediction_label,
                f"{probability:.2%}",
                risk_level,
                f"{CLASSIFICATION_THRESHOLD:.2f}",
            ],
        }
    )
    st.download_button(
        "Download Prediction Report",
        data=report.to_csv(index=False),
        file_name="customer_churn_prediction.csv",
        mime="text/csv",
    )


# =========================================================
# PAGE 6 — MODEL INSIGHTS
# =========================================================

def render_model_insights():
    page_header(
        "Model Insights",
        "Performance, feature importance and explainability of the final churn model",
        "Explainable AI",
    )

    cols = st.columns(5)
    cards = [
        ("Accuracy", "83.65%", "Final test set", "✅"),
        ("Balanced Accuracy", "77.11%", "Imbalance-aware", "⚖️"),
        ("Churn Recall", "66.09%", "At-risk customers detected", "🔎"),
        ("Churn F1", "62.20%", "Precision-recall balance", "📊"),
        ("ROC-AUC", "86.12%", "Ranking performance", "📈"),
    ]
    for col, card in zip(cols, cards):
        with col:
            kpi_card(*card)

    section_title("Model Configuration")
    c1, c2 = st.columns([1, 1])
    with c1:
        insight_card(
            "Final model: <strong>SMOTE Gradient Boosting</strong>. SMOTE was applied only to training data to improve minority-class detection."
        )
        insight_card(
            f"Decision threshold: <strong>{CLASSIFICATION_THRESHOLD:.2f}</strong>, selected on a validation subset to maximize churn F1-score without tuning directly on the test set."
        )
    with c2:
        insight_card(
            "Final confusion matrix: 1,404 true negatives, 189 false positives, 138 false negatives and 269 true positives."
        )
        insight_card(
            "The tuned model sacrifices some overall accuracy in exchange for substantially stronger churn recall than the original-data Gradient Boosting model."
        )

    section_title("Feature Importance")
    c3, c4 = st.columns(2)

    if not feature_importance_df.empty and {"Feature", "Importance"}.issubset(feature_importance_df.columns):
        fi = feature_importance_df.sort_values("Importance", ascending=False).head(10)
        fig_fi = px.bar(
            fi.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            color_discrete_sequence=[POWERBI_BLUE],
        )
        fig_fi = powerbi_layout(fig_fi, "Gradient Boosting Feature Importance")
        with c3:
            st.plotly_chart(fig_fi, use_container_width=True)
    else:
        with c3:
            st.info("feature_importance.csv was not found in outputs/tables.")

    if not permutation_importance_df.empty and {"Feature", "Importance"}.issubset(permutation_importance_df.columns):
        pi = permutation_importance_df.sort_values("Importance", ascending=False).head(10)
        fig_pi = px.bar(
            pi.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            color_discrete_sequence=[POWERBI_PURPLE],
        )
        fig_pi = powerbi_layout(fig_pi, "Permutation Importance")
        with c4:
            st.plotly_chart(fig_pi, use_container_width=True)
    else:
        with c4:
            st.info("permutation_importance.csv was not found in outputs/tables.")

    section_title("Explainability Summary")
    for text in [
        "Age is the strongest and most consistent driver across feature-importance, permutation-importance and SHAP analyses.",
        "Number of products is highly influential but has a strongly non-linear relationship with churn.",
        "Product engagement, activity status, geography and gender contribute meaningful risk information.",
        "Credit-card ownership and credit score contribute relatively little after stronger customer features are considered.",
    ]:
        insight_card(text)

    if not final_predictions_df.empty and "Churn_Probability" in final_predictions_df.columns:
        section_title("Test-Set Probability Distribution")
        fig_prob = px.histogram(
            final_predictions_df,
            x="Churn_Probability",
            nbins=30,
            color_discrete_sequence=[POWERBI_BLUE],
        )
        fig_prob.add_vline(
            x=CLASSIFICATION_THRESHOLD,
            line_dash="dash",
            line_color=POWERBI_RED,
            annotation_text=f"Threshold {CLASSIFICATION_THRESHOLD:.2f}",
        )
        fig_prob = powerbi_layout(fig_prob, "Distribution of Churn Probabilities")
        st.plotly_chart(fig_prob, use_container_width=True)


# =========================================================
# PAGE 7 — RECOMMENDATIONS
# =========================================================

def render_recommendations():
    page_header(
        "Recommendations",
        "Action-oriented retention priorities derived from EDA and model explainability",
        "Decision Support",
    )

    section_title("Strategic Retention Priorities")

    recommendations = [
        (
            "1. Prioritize the German customer base",
            "Germany records a 32.44% churn rate, approximately twice the rate observed in France and Spain. Regional retention campaigns and service-quality diagnostics should be prioritized there.",
        ),
        (
            "2. Target inactive customers",
            "Inactive customers churn at 26.85% versus 14.27% for active customers. Engagement campaigns, digital activation and relationship-manager outreach should focus on inactive accounts.",
        ),
        (
            "3. Protect the two-product relationship",
            "Customers with exactly two products show the strongest retention. Cross-sell suitable second products to single-product customers, but avoid assuming that more products always reduce churn.",
        ),
        (
            "4. Treat three- and four-product customers as a special review group",
            "Observed churn is exceptionally high among customers holding three or four products. Because these groups are small, investigate product complexity, service friction or unsuitable bundling before designing interventions.",
        ),
        (
            "5. Focus retention on ages 41–60",
            "Churn rises sharply in middle-to-older age groups and peaks at 56.21% among customers aged 51–60. Tailored service, loyalty and advisory propositions should be tested for these groups.",
        ),
        (
            "6. Protect high-value balances",
            "High-value churn is associated with approximately 58.71 million in balances. High-value customers above the model threshold should receive priority intervention because financial exposure is substantial.",
        ),
        (
            "7. Use risk-based outreach rather than blanket campaigns",
            "Use the probability bands Low, Medium, High and Critical Risk to prioritize retention resources. High and Critical customers should receive progressively stronger intervention.",
        ),
        (
            "8. Monitor the model after deployment",
            "Track churn recall, false positives, probability calibration and segment-level performance. Retrain when customer behaviour or the underlying data distribution changes materially.",
        ),
    ]

    for title, text in recommendations:
        recommendation_card(title, text)

    section_title("Suggested Operational Workflow")
    workflow = pd.DataFrame(
        {
            "Risk Level": ["Low", "Medium", "High", "Critical"],
            "Probability": [
                "< 30%",
                f"30% – < {CLASSIFICATION_THRESHOLD:.0%}",
                f"{CLASSIFICATION_THRESHOLD:.0%} – < 75%",
                "≥ 75%",
            ],
            "Recommended Action": [
                "Standard service and monitoring",
                "Targeted engagement / product review",
                "Proactive retention outreach",
                "Immediate relationship-manager intervention",
            ],
        }
    )
    st.dataframe(workflow, use_container_width=True, hide_index=True)


# =========================================================
# PAGE DISPATCH
# =========================================================

PAGE_RENDERERS = {
    "Executive Overview": render_executive_overview,
    "Customer Segmentation": render_customer_segmentation,
    "Churn Analytics": render_churn_analytics,
    "High-Value Customers": render_high_value_customers,
    "Customer Prediction": render_customer_prediction,
    "Model Insights": render_model_insights,
    "Recommendations": render_recommendations,
}

PAGE_RENDERERS[page]()