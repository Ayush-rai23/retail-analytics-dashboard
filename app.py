import streamlit as st
import pandas as pd
import plotly.express as px
from data_processing import (
    load_uploaded_file,
    validate_dataset,
    preprocess_dataset,
    compute_statistical_summary,
    calculate_kpis,
)

st.set_page_config(
    page_title="Enterprise Retail Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_kpi_cards(kpis: dict) -> None:
    """Display the five required dynamically updating KPI metrics."""
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Sales", f"${kpis['total_sales']:,.2f}")
    col2.metric("Total Profit", f"${kpis['total_profit']:,.2f}")
    col3.metric("Total Orders", f"{kpis['total_orders']:,}")
    col4.metric("Total Quantity Sold", f"{int(kpis['total_quantity']):,}")
    col5.metric("Profit Margin (%)", f"{kpis['profit_margin']:.2f}%")


def apply_interactive_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filters (Date Range, Region, Category, Segment) and filter dataframe."""
    st.sidebar.header("🔍 Interactive Filters")
    filtered = df.copy()

    # Filter 1: Date Range
    if "Order Date" in filtered.columns and pd.api.types.is_datetime64_any_dtype(filtered["Order Date"]):
        min_date = filtered["Order Date"].min().date()
        max_date = filtered["Order Date"].max().date()
        date_range = st.sidebar.date_input(
            "Order Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            filtered = filtered[(filtered["Order Date"] >= start_dt) & (filtered["Order Date"] <= end_dt)]

    # Filter 2: Region
    if "Region" in filtered.columns:
        regions = sorted(filtered["Region"].dropna().unique().tolist())
        selected_regions = st.sidebar.multiselect("Region", options=regions, default=regions)
        if selected_regions:
            filtered = filtered[filtered["Region"].isin(selected_regions)]

    # Filter 3: Category
    if "Category" in filtered.columns:
        categories = sorted(filtered["Category"].dropna().unique().tolist())
        selected_categories = st.sidebar.multiselect("Category", options=categories, default=categories)
        if selected_categories:
            filtered = filtered[filtered["Category"].isin(selected_categories)]

    # Filter 4: Customer Segment
    if "Segment" in filtered.columns:
        segments = sorted(filtered["Segment"].dropna().unique().tolist())
        selected_segments = st.sidebar.multiselect("Segment", options=segments, default=segments)
        if selected_segments:
            filtered = filtered[filtered["Segment"].isin(selected_segments)]

    return filtered


def render_charts(df: pd.DataFrame) -> None:
    """Create four interactive Plotly visualizations."""
    if df.empty:
        st.warning("No records match the current filter criteria. Adjust your sidebar filters.")
        return

    row1_col1, row1_col2 = st.columns(2)

    # Chart 1: Monthly Sales & Profit Trend (Line Chart)
    with row1_col1:
        st.subheader("📈 Monthly Sales & Profit Trend")
        if "Order Date" in df.columns and {"Sales", "Profit"}.issubset(df.columns):
            trend_df = (
                df.assign(Month=df["Order Date"].dt.to_period("M").dt.to_timestamp())
                .groupby("Month", as_index=False)[["Sales", "Profit"]]
                .sum()
                .sort_values("Month")
            )
            fig_trend = px.line(
                trend_df,
                x="Month",
                y=["Sales", "Profit"],
                markers=True,
                labels={"value": "Amount ($)", "variable": "Metric"},
                color_discrete_map={"Sales": "#1f77b4", "Profit": "#2ca02c"},
            )
            fig_trend.update_layout(hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_trend, use_container_width=True)

    # Chart 2: Sales & Profit by Category and Sub-Category (Grouped Bar Chart)
    with row1_col2:
        st.subheader("📊 Sales & Profit by Category")
        if {"Category", "Sales", "Profit"}.issubset(df.columns):
            cat_df = df.groupby("Category", as_index=False)[["Sales", "Profit"]].sum()
            fig_cat = px.bar(
                cat_df,
                x="Category",
                y=["Sales", "Profit"],
                barmode="group",
                text_auto=".2s",
                labels={"value": "Amount ($)", "variable": "Metric"},
                color_discrete_map={"Sales": "#3b82f6", "Profit": "#10b981"},
            )
            fig_cat.update_layout(margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_cat, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)

    # Chart 3: Regional Sales Share (Donut Chart)
    with row2_col1:
        st.subheader("🗺️ Sales Distribution by Region")
        if {"Region", "Sales"}.issubset(df.columns):
            reg_df = df.groupby("Region", as_index=False)["Sales"].sum()
            fig_reg = px.pie(
                reg_df,
                names="Region",
                values="Sales",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_reg.update_traces(textposition="inside", textinfo="percent+label")
            fig_reg.update_layout(margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_reg, use_container_width=True)

    # Chart 4: Discount vs. Profit Scatter Analysis
    with row2_col2:
        st.subheader("🎯 Discount vs. Profit Impact")
        if {"Discount", "Profit", "Sales"}.issubset(df.columns):
            color_col = "Category" if "Category" in df.columns else None
            hover_cols = [c for c in ["Product Name", "Sub-Category", "Region"] if c in df.columns]
            fig_scatter = px.scatter(
                df,
                x="Discount",
                y="Profit",
                size="Sales",
                color=color_col,
                hover_data=hover_cols,
                labels={"Discount": "Discount Rate", "Profit": "Profit ($)"},
            )
            fig_scatter.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
            fig_scatter.update_layout(margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_scatter, use_container_width=True)


def main() -> None:
    st.title("📊 Sales & Profit Analytics Application")
    st.caption("Upload CSV, Excel (.xlsx), or JSON (.json) datasets for automated validation, cleaning, and interactive BI analysis.")

    st.sidebar.header("📂 1. Data Upload & Cleaning")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Dataset",
        type=["csv", "xlsx", "json"],
        help="Supported formats: CSV (.csv), Excel (.xlsx), JSON (.json)",
    )

    missing_strategy = st.sidebar.selectbox(
        "Missing Value Strategy",
        options=["Impute (Median / Mode)", "Drop Rows with Missing Values"],
    )
    remove_dups = st.sidebar.checkbox("Remove Duplicate Records", value=True)
    cap_outliers = st.sidebar.checkbox("Sanitize Outliers & Invalid Entries", value=True)

    if uploaded_file is None:
        st.info("👈 Upload a `.csv`, `.xlsx`, or `.json` dataset from the sidebar to begin.")
        return

    try:
        raw_df = load_uploaded_file(uploaded_file)
        validation_report = validate_dataset(raw_df)
    except Exception as exc:
        st.error(f"File loading or validation failed: {exc}")
        return

    tab_dashboard, tab_validation, tab_preprocessing = st.tabs([
        "📈 Interactive Dashboard",
        "🔍 Data Validation Report",
        "🧹 Preprocessing & Statistics",
    ])

    # Preprocess dataset
    try:
        cleaned_df, audit_log = preprocess_dataset(
            raw_df,
            missing_strategy=missing_strategy,
            remove_duplicates=remove_dups,
            cap_outliers=cap_outliers,
        )
    except Exception as exc:
        st.error(f"Preprocessing error: {exc}")
        return

    # Tab 1: Interactive Filters, KPIs & Visualizations
    with tab_dashboard:
        filtered_df = apply_interactive_filters(cleaned_df)
        kpis = calculate_kpis(filtered_df)
        render_kpi_cards(kpis)
        st.divider()
        render_charts(filtered_df)

    # Tab 2: Raw Data Validation
    with tab_validation:
        st.subheader("Raw Dataset Quality Inspection")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Rows", f"{validation_report['rows']:,}")
        v2.metric("Columns", f"{validation_report['columns']:,}")
        v3.metric("Total Missing Cells", f"{validation_report['total_missing']:,}")
        v4.metric("Duplicate Rows", f"{validation_report['duplicate_rows']:,}")

        st.markdown("**Column Schema, Data Types & Missing Value Breakdown**")
        st.dataframe(validation_report["schema_summary"], use_container_width=True)

        st.markdown("**Raw Data Preview (First 20 Rows)**")
        st.dataframe(raw_df.head(20), use_container_width=True)

    # Tab 3: Preprocessing Audit & Statistical Summaries
    with tab_preprocessing:
        st.subheader("Data Cleaning Audit Summary")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Initial Rows", audit_log["initial_rows"])
        a2.metric("Final Clean Rows", len(cleaned_df))
        a3.metric("Duplicates Removed", audit_log["duplicates_removed"])
        a4.metric("Anomalies Sanitized", audit_log["anomalies_corrected"])

        st.subheader("Descriptive Statistical Summary (Cleaned Numeric Features)")
        stats_df = compute_statistical_summary(cleaned_df)
        st.dataframe(stats_df, use_container_width=True)

        st.subheader("Cleaned Dataset Preview & Export")
        st.dataframe(cleaned_df.head(20), use_container_width=True)
        csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Cleaned Dataset (CSV)",
            data=csv_bytes,
            file_name="cleaned_retail_dataset.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()