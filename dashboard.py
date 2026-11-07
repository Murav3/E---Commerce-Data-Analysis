
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Olist E-Commerce Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# FILE PATHS
# =====================================================

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

FACT_FILE = OUT / "cleaned_order_items.csv"
ORDERS_FILE = OUT / "order_level_analysis.csv"


# =====================================================
# DASHBOARD CSS
# =====================================================

st.markdown("""
<style>

:root {
    --navy: #14213D;
    --ink: #172B4D;
    --muted: #64748B;
    --teal: #168AAD;
    --canvas: #F3F5F9;
    --line: #E2E8F0;
}

/* Main background */

.stApp,
[data-testid="stAppViewContainer"] {
    background-color: #F3F5F9 !important;
}

/* Main text */

.stApp p,
.stApp label,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
    color: #172B4D;
}

/* Sidebar */

[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background-color: #14213D !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}

/* Sidebar dropdown */

[data-testid="stSidebar"] [data-baseweb="select"] {
    background-color: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] input {
    color: #172B4D !important;
    -webkit-text-fill-color: #172B4D !important;
}

[data-testid="stSidebar"] [data-baseweb="tag"] {
    background-color: #DCEEF3 !important;
}

[data-testid="stSidebar"] [data-baseweb="tag"] * {
    color: #14213D !important;
}

/* Header */

.main-title {
    color: #14213D !important;
    font-size: 31px;
    font-weight: 800;
    margin-bottom: 5px;
}

.sub-title {
    color: #64748B !important;
    font-size: 14px;
    margin-bottom: 20px;
}

/* Section heading */

.section-title {
    color: #14213D !important;
    font-size: 20px;
    font-weight: 800;
    margin: 18px 0 12px 0;
}

/* KPI cards */

.kpi-card {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0;
    border-top: 4px solid #168AAD;
    border-radius: 12px;
    padding: 17px;
    min-height: 112px;
    box-shadow: 0 3px 10px rgba(20,33,61,0.05);
}

.kpi-label {
    color: #64748B !important;
    font-size: 11px;
    font-weight: 750;
    letter-spacing: 0.5px;
    margin-bottom: 9px;
}

.kpi-value {
    color: #14213D !important;
    font-size: 23px;
    font-weight: 800;
    overflow-wrap: anywhere;
}

.kpi-note {
    color: #8492A6 !important;
    font-size: 11px;
    margin-top: 7px;
}

/* Chart cards */

[data-testid="stPlotlyChart"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 8px;
    box-shadow: 0 3px 10px rgba(20,33,61,0.04);
}

/* Tables */

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background-color: #FFFFFF !important;
}

/* Expanders */

[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
}

/* Buttons */

.stButton button,
.stDownloadButton button {
    background-color: #168AAD !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}

.stButton button:hover,
.stDownloadButton button:hover {
    background-color: #116B86 !important;
    color: #FFFFFF !important;
}

/* Divider */

hr {
    border-color: #DCE3EC !important;
}

#MainMenu,
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    if not FACT_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {FACT_FILE}"
        )

    if not ORDERS_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {ORDERS_FILE}"
        )

    fact = pd.read_csv(FACT_FILE)
    orders = pd.read_csv(ORDERS_FILE)

    # Standardise category column

    if "category" not in fact.columns:

        if "product_category_name_english" in fact.columns:
            fact = fact.rename(
                columns={
                    "product_category_name_english": "category"
                }
            )

        elif "product_category_name" in fact.columns:
            fact = fact.rename(
                columns={
                    "product_category_name": "category"
                }
            )

    # Handle date column names

    if (
        "order_purchase_timestamp" not in fact.columns
        and "order_date" in fact.columns
    ):
        fact["order_purchase_timestamp"] = fact["order_date"]

    if (
        "order_date" not in orders.columns
        and "order_purchase_timestamp" in orders.columns
    ):
        orders["order_date"] = orders[
            "order_purchase_timestamp"
        ]

    # Required columns

    required_fact = [
        "order_id",
        "price",
        "customer_state",
        "order_purchase_timestamp",
        "category"
    ]

    required_orders = [
        "order_id",
        "order_value",
        "review_score",
        "order_date"
    ]

    missing_fact = [
        c for c in required_fact
        if c not in fact.columns
    ]

    missing_orders = [
        c for c in required_orders
        if c not in orders.columns
    ]

    if missing_fact:
        raise ValueError(
            f"Missing columns in cleaned_order_items.csv: "
            f"{missing_fact}. Available: {fact.columns.tolist()}"
        )

    if missing_orders:
        raise ValueError(
            f"Missing columns in order_level_analysis.csv: "
            f"{missing_orders}. Available: {orders.columns.tolist()}"
        )

    # Numeric conversion

    fact["price"] = pd.to_numeric(
        fact["price"], errors="coerce"
    ).fillna(0)

    orders["order_value"] = pd.to_numeric(
        orders["order_value"], errors="coerce"
    )

    orders["review_score"] = pd.to_numeric(
        orders["review_score"], errors="coerce"
    )

    # Date conversion

    fact["order_purchase_timestamp"] = pd.to_datetime(
        fact["order_purchase_timestamp"],
        errors="coerce"
    )

    orders["order_date"] = pd.to_datetime(
        orders["order_date"],
        errors="coerce"
    )

    fact = fact.dropna(
        subset=["order_purchase_timestamp"]
    ).copy()

    orders = orders.dropna(
        subset=["order_date"]
    ).copy()

    # Clean text

    fact["category"] = (
        fact["category"]
        .fillna("Unknown")
        .astype(str)
        .str.replace("_", " ", regex=False)
        .str.title()
    )

    fact["customer_state"] = (
        fact["customer_state"]
        .fillna("Unknown")
        .astype(str)
        .str.upper()
    )

    # Create month column

    fact["month"] = (
        fact["order_purchase_timestamp"]
        .dt.to_period("M")
        .astype(str)
    )

    orders["month"] = (
        orders["order_date"]
        .dt.to_period("M")
        .astype(str)
    )

    return fact, orders


try:
    fact, orders = load_data()

except Exception as e:
    st.error(f"Dashboard could not load the data: {e}")
    st.stop()


# =====================================================
# SIDEBAR FILTERS
# =====================================================

st.sidebar.markdown("## 📊 Olist Analytics")
st.sidebar.caption("E-Commerce Business Intelligence")
st.sidebar.markdown("---")
st.sidebar.markdown("### Dashboard Filters")

months = sorted(fact["month"].unique().tolist())
categories = sorted(fact["category"].unique().tolist())
states = sorted(fact["customer_state"].unique().tolist())


# Initialise filter values

if "month_filter" not in st.session_state:
    st.session_state["month_filter"] = months.copy()

if "category_filter" not in st.session_state:
    st.session_state["category_filter"] = categories.copy()

if "state_filter" not in st.session_state:
    st.session_state["state_filter"] = states.copy()


def reset_filters():

    st.session_state["month_filter"] = months.copy()
    st.session_state["category_filter"] = categories.copy()
    st.session_state["state_filter"] = states.copy()


st.sidebar.multiselect(
    "Select Month",
    months,
    key="month_filter"
)

st.sidebar.multiselect(
    "Select Product Category",
    categories,
    key="category_filter"
)

st.sidebar.multiselect(
    "Select Customer State",
    states,
    key="state_filter"
)

st.sidebar.button(
    "↺ Reset All Filters",
    use_container_width=True,
    on_click=reset_filters
)

st.sidebar.markdown("---")
st.sidebar.caption("Source: Olist Brazilian E-Commerce Dataset")


selected_months = st.session_state["month_filter"]
selected_categories = st.session_state["category_filter"]
selected_states = st.session_state["state_filter"]


# =====================================================
# FILTER DATA
# =====================================================

filtered_fact = fact[
    fact["month"].isin(selected_months)
    & fact["category"].isin(selected_categories)
    & fact["customer_state"].isin(selected_states)
].copy()

selected_order_ids = filtered_fact[
    "order_id"
].dropna().unique()

filtered_orders = orders[
    orders["order_id"].isin(selected_order_ids)
    & orders["month"].isin(selected_months)
].copy()


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def money(value):
    return f"R$ {value:,.2f}"


def kpi_card(label, value, note):

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def style_chart(fig, height=360):

    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",

        font=dict(
            family="Arial",
            size=12,
            color="#172B4D"
        ),

        title=dict(
            font=dict(
                size=16,
                color="#172B4D"
            ),
            x=0.03
        ),

        margin=dict(
            l=35,
            r=35,
            t=65,
            b=45
        ),

        legend=dict(
            title=dict(
                font=dict(color="#172B4D")
            ),
            font=dict(
                color="#344054",
                size=11
            ),
            bgcolor="#FFFFFF"
        ),

        hoverlabel=dict(
            bgcolor="#FFFFFF",
            font=dict(
                color="#172B4D",
                size=12
            )
        )
    )

    fig.update_xaxes(
        title_font=dict(color="#172B4D"),
        tickfont=dict(color="#344054"),
        showgrid=False,
        zeroline=False,
        linecolor="#DCE3EC"
    )

    fig.update_yaxes(
        title_font=dict(color="#172B4D"),
        tickfont=dict(color="#344054"),
        showgrid=True,
        gridcolor="#E9EEF4",
        zeroline=False
    )

    return fig


# =====================================================
# DASHBOARD HEADER
# =====================================================

st.markdown(
    '<div class="main-title">E-Commerce Sales Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Business performance overview | Revenue, orders, products and customer experience'
    '</div>',
    unsafe_allow_html=True
)


# =====================================================
# KPI CALCULATIONS
# =====================================================

total_revenue = filtered_fact["price"].sum()

total_orders = filtered_fact["order_id"].nunique()

items_sold = len(filtered_fact)

avg_order_value = filtered_orders["order_value"].mean()

avg_review = filtered_orders["review_score"].mean()


# =====================================================
# EXECUTIVE OVERVIEW
# =====================================================

st.markdown(
    '<div class="section-title">Executive Overview</div>',
    unsafe_allow_html=True
)

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    kpi_card(
        "TOTAL REVENUE",
        money(total_revenue),
        "Sum of item prices"
    )

with k2:
    kpi_card(
        "TOTAL ORDERS",
        f"{total_orders:,}",
        "Unique orders"
    )

with k3:
    kpi_card(
        "ITEMS SOLD",
        f"{items_sold:,}",
        "Total order items"
    )

with k4:
    kpi_card(
        "AVERAGE ORDER VALUE",
        money(avg_order_value)
        if pd.notna(avg_order_value)
        else "R$ 0.00",
        "Mean order-level value"
    )

with k5:
    kpi_card(
        "AVERAGE REVIEW",
        f"{avg_review:.2f} / 5"
        if pd.notna(avg_review)
        else "N/A",
        "Mean customer review score"
    )


# =====================================================
# SALES PERFORMANCE
# =====================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">Sales Performance</div>',
    unsafe_allow_html=True
)

left, right = st.columns([1.45, 1])


# Monthly Revenue

with left:

    monthly_revenue = (
        filtered_fact
        .groupby("month", as_index=False)["price"]
        .sum()
        .sort_values("month")
    )

    monthly_fig = px.line(
        monthly_revenue,
        x="month",
        y="price",
        markers=True,
        title="Monthly Revenue Trend",
        labels={
            "month": "Month",
            "price": "Revenue"
        }
    )

    monthly_fig.update_traces(
        line=dict(
            color="#168AAD",
            width=3
        ),
        marker=dict(
            size=7,
            color="#168AAD"
        ),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Revenue: R$ %{y:,.2f}"
            "<extra></extra>"
        )
    )

    monthly_fig.update_yaxes(
        tickprefix="R$ ",
        tickformat=",.0f"
    )

    st.plotly_chart(
        style_chart(monthly_fig, 370),
        use_container_width=True,
        config={"displayModeBar": False}
    )


# Top 10 Categories

with right:

    category_revenue = (
        filtered_fact
        .groupby("category", as_index=False)["price"]
        .sum()
        .nlargest(10, "price")
        .sort_values("price")
    )

    category_fig = px.bar(
        category_revenue,
        x="price",
        y="category",
        orientation="h",
        title="Top 10 Product Categories",
        labels={
            "price": "Revenue (R$)",
            "category": ""
        },
        color_discrete_sequence=["#299DAD"]
    )

    category_fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Revenue: R$ %{x:,.2f}"
            "<extra></extra>"
        )
    )

    category_fig.update_xaxes(
        tickprefix="R$ ",
        tickformat=",.0f"
    )

    st.plotly_chart(
        style_chart(category_fig, 370),
        use_container_width=True,
        config={"displayModeBar": False}
    )


# =====================================================
# REGIONAL SALES
# =====================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">Regional Sales Analysis</div>',
    unsafe_allow_html=True
)

state_revenue = (
    filtered_fact
    .groupby("customer_state", as_index=False)["price"]
    .sum()
    .sort_values("price", ascending=False)
)

reg_left, reg_right = st.columns([1.2, 1])


# Top 10 States Bar Chart

with reg_left:

    top10_states = (
        state_revenue
        .head(10)
        .sort_values("price")
    )

    state_fig = px.bar(
        top10_states,
        x="price",
        y="customer_state",
        orientation="h",
        title="Top 10 States by Revenue",
        labels={
            "price": "Revenue (R$)",
            "customer_state": "State"
        },
        color="price",
        color_continuous_scale=[
            "#B9E2EA",
            "#168AAD",
            "#14213D"
        ]
    )

    state_fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Revenue: R$ %{x:,.2f}"
            "<extra></extra>"
        )
    )

    state_fig.update_layout(
        coloraxis_showscale=False
    )

    state_fig.update_xaxes(
        tickprefix="R$ ",
        tickformat=",.0f"
    )

    st.plotly_chart(
        style_chart(state_fig, 410),
        use_container_width=True,
        config={"displayModeBar": False}
    )


# =====================================================
# REVENUE CONTRIBUTION DONUT CHART
# =====================================================

with reg_right:

    # Top 8 states

    contribution = state_revenue.head(8).copy()

    # All remaining states

    others_value = state_revenue.iloc[8:]["price"].sum()

    if others_value > 0:

        contribution = pd.concat(
            [
                contribution,
                pd.DataFrame({
                    "customer_state": ["Others"],
                    "price": [others_value]
                })
            ],
            ignore_index=True
        )

    total_share = contribution["price"].sum()

    if total_share > 0:

        colors = [
            "#176B87",
            "#299DAD",
            "#70C1B3",
            "#F4A261",
            "#E76F51",
            "#6477B9",
            "#9B72CF",
            "#A5B4C8",
            "#D5DCE5"
        ]

        pie = go.Figure(
            data=[
                go.Pie(
                    labels=contribution["customer_state"],
                    values=contribution["price"],
                    hole=0.60,
                    sort=False,

                    marker=dict(
                        colors=colors[:len(contribution)],
                        line=dict(
                            color="#FFFFFF",
                            width=2
                        )
                    ),

                    textinfo="percent",
                    textposition="inside",

                    insidetextfont=dict(
                        color="#FFFFFF",
                        size=11
                    ),

                    hovertemplate=(
                        "<b>%{label}</b><br>"
                        "Revenue: R$ %{value:,.2f}<br>"
                        "Contribution: %{percent}"
                        "<extra></extra>"
                    )
                )
            ]
        )

        pie.update_layout(
            template="plotly_white",
            height=410,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",

            font=dict(
                family="Arial",
                size=11,
                color="#172B4D"
            ),

            title=dict(
                text="Top 8 States + Others",
                x=0.03,
                font=dict(
                    size=16,
                    color="#172B4D"
                )
            ),

            margin=dict(
                l=10,
                r=125,
                t=65,
                b=20
            ),

            showlegend=True,

            legend=dict(
                title=dict(text="State"),
                orientation="v",
                x=1.02,
                xanchor="left",
                y=0.5,
                yanchor="middle",
                font=dict(
                    size=11,
                    color="#172B4D"
                ),
                bgcolor="#FFFFFF"
            ),

            annotations=[
                dict(
                    text="Revenue<br>Share",
                    x=0.43,
                    y=0.5,
                    showarrow=False,
                    font=dict(
                        size=14,
                        color="#14213D"
                    )
                )
            ]
        )

        st.plotly_chart(
            pie,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    else:
        st.info(
            "No state revenue is available for the selected filters."
        )


# =====================================================
# CUSTOMER SATISFACTION
# =====================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">Customer Satisfaction</div>',
    unsafe_allow_html=True
)

review_left, review_right = st.columns(2)


# Customer Review Distribution

with review_left:

    review_data = (
        filtered_orders["review_score"]
        .dropna()
        .value_counts()
        .sort_index()
        .rename_axis("review_score")
        .reset_index(name="review_count")
    )

    if not review_data.empty:

        review_fig = px.pie(
            review_data,
            names="review_score",
            values="review_count",
            hole=0.55,
            title="Customer Review Distribution",
            color="review_score",
            color_discrete_sequence=[
                "#E76F51",
                "#F4A261",
                "#E9C46A",
                "#70C1B3",
                "#176B87"
            ]
        )

        review_fig.update_traces(
            textinfo="percent",
            hovertemplate=(
                "Rating: %{label}<br>"
                "Reviews: %{value:,}<br>"
                "Share: %{percent}"
                "<extra></extra>"
            )
        )

        st.plotly_chart(
            style_chart(review_fig, 360),
            use_container_width=True,
            config={"displayModeBar": False}
        )

    else:
        st.info(
            "No review data is available for the selected filters."
        )


# Monthly Average Order Value

with review_right:

    monthly_aov = (
        filtered_orders
        .groupby("month", as_index=False)["order_value"]
        .mean()
        .sort_values("month")
    )

    if not monthly_aov.empty:

        aov_fig = px.bar(
            monthly_aov,
            x="month",
            y="order_value",
            title="Monthly Average Order Value",
            labels={
                "month": "Month",
                "order_value": "Average Order Value (R$)"
            },
            color_discrete_sequence=["#6477B9"]
        )

        aov_fig.update_traces(
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Average Order Value: R$ %{y:,.2f}"
                "<extra></extra>"
            )
        )

        aov_fig.update_yaxes(
            tickprefix="R$ ",
            tickformat=",.0f"
        )

        st.plotly_chart(
            style_chart(aov_fig, 360),
            use_container_width=True,
            config={"displayModeBar": False}
        )

    else:
        st.info(
            "No order-value data is available for the selected filters."
        )


# =====================================================
# CATEGORY PERFORMANCE HEATMAP
# =====================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">Category Performance Over Time</div>',
    unsafe_allow_html=True
)

heatmap_data = (
    filtered_fact
    .groupby(
        ["category", "month"],
        as_index=False
    )["price"]
    .sum()
)

if not heatmap_data.empty:

    heatmap_pivot = heatmap_data.pivot(
        index="category",
        columns="month",
        values="price"
    ).fillna(0)

    top_category_names = (
        heatmap_pivot
        .sum(axis=1)
        .nlargest(15)
        .index
    )

    heatmap_pivot = heatmap_pivot.loc[
        top_category_names
    ]

    heatmap_fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_pivot.values,
            x=heatmap_pivot.columns.tolist(),
            y=heatmap_pivot.index.tolist(),

            colorscale="Blues",

            colorbar=dict(
                title=dict(
                    text="Revenue (R$)",
                    font=dict(color="#172B4D")
                ),
                tickfont=dict(color="#172B4D")
            ),

            hovertemplate=(
                "Category: %{y}<br>"
                "Month: %{x}<br>"
                "Revenue: R$ %{z:,.2f}"
                "<extra></extra>"
            )
        )
    )

    heatmap_fig.update_layout(
        template="plotly_white",

        title=dict(
            text="Monthly Revenue by Product Category",
            x=0.03,
            font=dict(
                size=16,
                color="#172B4D"
            )
        ),

        height=520,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",

        font=dict(
            family="Arial",
            size=11,
            color="#172B4D"
        ),

        margin=dict(
            l=20,
            r=25,
            t=65,
            b=45
        )
    )

    heatmap_fig.update_xaxes(
        tickfont=dict(color="#344054"),
        title_text="Month",
        showgrid=False
    )

    heatmap_fig.update_yaxes(
        tickfont=dict(color="#344054"),
        title_text="Category",
        showgrid=False
    )

    st.plotly_chart(
        heatmap_fig,
        use_container_width=True,
        config={"displayModeBar": False}
    )

else:
    st.info(
        "No category/month data is available for the selected filters."
    )


# =====================================================
# DETAILED SALES RECORDS
# =====================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">Detailed Sales Records</div>',
    unsafe_allow_html=True
)

with st.expander(
    "View filtered sales data",
    expanded=False
):

    st.dataframe(
        filtered_fact,
        use_container_width=True,
        hide_index=True
    )

    csv_bytes = filtered_fact.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇ Download Filtered Data",
        data=csv_bytes,
        file_name="olist_filtered_sales.csv",
        mime="text/csv"
    )


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Olist E-Commerce Analytics Dashboard | "
    "Built with Streamlit, Pandas and Plotly"
)
