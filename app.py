"""
app.py — ByteMind Streamlit entry point
---------------------------------------
Run with:
    streamlit run app.py
"""

import streamlit as st

from bytemind import analysis, data_health, file_converter, investigation

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ByteMind — Data Detective",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session-state: shared dataset
# ---------------------------------------------------------------------------
# Any page can read/write st.session_state["dataset"] (pd.DataFrame | None)
# and st.session_state["dataset_name"] (str).
# This lets the user upload once and use the dataset across all features.
if "dataset" not in st.session_state:
    st.session_state["dataset"] = None
if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = ""

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://via.placeholder.com/200x60/1a1a2e/ffffff?text=ByteMind",
        use_container_width=True,
    )
    st.markdown("### Navigation")
    page = st.radio(
        label="Go to",
        options=[
            "🏠 Home",
            "📂 Upload & Convert",
            "🏥 Data Health",
            "📊 Analysis",
            "🔍 Investigation",
        ],
        label_visibility="collapsed",
    )

    # Show active dataset indicator
    if st.session_state["dataset"] is not None:
        df_active = st.session_state["dataset"]
        st.divider()
        st.markdown("**Active dataset**")
        st.caption(
            f"📄 {st.session_state['dataset_name']}\n\n"
            f"{len(df_active):,} rows × {len(df_active.columns)} cols"
        )
    else:
        st.divider()
        st.caption("No dataset loaded yet.")

    st.divider()
    st.caption("ByteMind v0.2.0 — IBM Bob 2.0 Hackathon")


# ---------------------------------------------------------------------------
# Shared upload widget (used on pages that need a dataset)
# ---------------------------------------------------------------------------
def _shared_upload_widget(context_label: str = "this feature") -> bool:
    """
    Render a dataset-loading UI that populates session state.

    Behaviour
    ---------
    * If a dataset is already loaded in session state, show a compact
      "Dataset loaded" banner and a "Replace dataset" expander — no
      file-uploader widget is shown at the top level.  Returns True.
    * If no dataset is loaded, show the file-uploader.  Returns True
      after a successful upload, False if nothing is loaded.

    This means navigating away and back never shows an empty uploader
    when a dataset is already available.
    """
    dataset_present = st.session_state["dataset"] is not None

    if dataset_present:
        # Show a clear "already loaded" indicator
        df_current = st.session_state["dataset"]
        name_current = st.session_state["dataset_name"]
        st.success(
            f"✅ Dataset loaded: **{name_current}** — "
            f"{len(df_current):,} rows × {len(df_current.columns)} columns."
        )
        with st.expander("🔄 Replace dataset", expanded=False):
            uploaded = st.file_uploader(
                "Upload a new file to replace the current dataset",
                type=["csv", "xlsx", "json"],
                help="Supported formats: CSV, Excel (.xlsx), JSON",
                key=f"uploader_replace_{context_label}",
            )
            if uploaded is not None:
                ext = uploaded.name.rsplit(".", 1)[-1].lower()
                try:
                    df = file_converter.load_file(uploaded, ext)
                    st.session_state["dataset"] = df
                    st.session_state["dataset_name"] = uploaded.name
                    st.success(
                        f"Replaced with **{uploaded.name}** — "
                        f"{len(df):,} rows × {len(df.columns)} columns."
                    )
                except ValueError as err:
                    st.error(f"Could not load the file: {err}")
        return True

    # No dataset in session state — show the primary uploader
    uploaded = st.file_uploader(
        "Upload a dataset to get started",
        type=["csv", "xlsx", "json"],
        help="Supported formats: CSV, Excel (.xlsx), JSON",
        key=f"uploader_{context_label}",
    )

    if uploaded is not None:
        ext = uploaded.name.rsplit(".", 1)[-1].lower()
        try:
            df = file_converter.load_file(uploaded, ext)
            st.session_state["dataset"] = df
            st.session_state["dataset_name"] = uploaded.name
            st.success(
                f"Loaded **{uploaded.name}** — "
                f"{len(df):,} rows × {len(df.columns)} columns."
            )
            return True
        except ValueError as err:
            st.error(f"Could not load the file: {err}")
            return False

    st.info(
        f"Upload a CSV, Excel (.xlsx), or JSON file above to use {context_label}. "
        "Once loaded, the dataset is shared across all ByteMind features — "
        "you will not need to re-upload it on other pages."
    )
    return False


# ===========================================================================
# HOME PAGE
# ===========================================================================
if page == "🏠 Home":
    st.markdown(
        """
        <h1 style='text-align:center; color:#1a1a2e;'>
            🔍 ByteMind
        </h1>
        <p style='text-align:center; font-size:1.2rem; color:#4a4a6a;'>
            <strong>Data Detective &amp; Investigation Assistant</strong><br>
            For Junior Data Analysts &amp; Small-Business Owners
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.subheader("The Problem We Solve")
    st.write(
        "Working with real-world datasets is messy. "
        "Data arrives in different formats, hides quality issues, and rarely "
        "announces its own interesting patterns. "
        "ByteMind acts as your personal data detective — converting, checking, "
        "analysing, and surfacing the findings that actually deserve your attention."
    )
    st.divider()

    st.subheader("How ByteMind Works")
    workflow_steps = [
        ("📂", "Upload Dataset", "CSV, Excel (.xlsx), or JSON"),
        ("🔄", "Format Conversion", "Convert between supported formats"),
        ("🏥", "Data Health Check", "Missing values, duplicates, outliers, types"),
        ("📊", "Automatic Analysis", "Statistics, correlations, trends, charts"),
        ("💡", "Important Findings", "Surfacing what genuinely matters"),
        ("🔍", "Investigation Suggestions", "Concrete follow-up questions"),
        ("🤖", "AI Explanation", "Plain-language or technical narrative"),
    ]
    cols = st.columns(len(workflow_steps))
    for col, (icon, title, desc) in zip(cols, workflow_steps):
        with col:
            st.markdown(
                f"""
                <div style='
                    background:#f0f2f6; border-radius:10px; padding:14px 10px;
                    text-align:center; height:140px; display:flex;
                    flex-direction:column; align-items:center; justify-content:center;
                '>
                    <div style='font-size:2rem;'>{icon}</div>
                    <div style='font-weight:700; font-size:0.85rem; margin-top:6px;'>{title}</div>
                    <div style='font-size:0.73rem; color:#4a4a6a; margin-top:4px;'>{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown(
        "<p style='text-align:center; font-size:0.8rem; color:#888; margin-top:8px;'>"
        "Each step feeds into the next ↓</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.subheader("Who Is ByteMind For?")
    col_a, col_b = st.columns(2)
    with col_a:
        st.info(
            "**📈 Junior Data Analysts**\n\n"
            "Get technical health reports, correlation matrices, and statistical "
            "summaries — plus investigation leads you can take straight into your "
            "next analysis sprint."
        )
    with col_b:
        st.info(
            "**🏪 Small-Business Owners**\n\n"
            "No statistics degree required. ByteMind surfaces plain-language "
            "findings and business questions so you can make confident decisions "
            "from your own data."
        )
    st.divider()

    st.subheader("Feature Roadmap — MVP Status")
    status_data = {
        "Feature": [
            "File Upload & Conversion",
            "Data Health Check",
            "Automatic Analysis",
            "Investigation Engine",
        ],
        "Status": [
            "✅ Complete",
            "✅ Complete",
            "✅ Complete",
            "✅ Complete",
        ],
        "Description": [
            "CSV, Excel, JSON upload, preview, conversion & download",
            "Overview, missing values, duplicates, outliers, health score",
            "Descriptive stats, correlations, trends, top/bottom, auto-findings",
            "Data-derived investigation leads with follow-up questions",
        ],
    }
    st.table(status_data)
    st.caption(
        "Navigate to any live feature in the sidebar to get started."
    )


# ===========================================================================
# UPLOAD & CONVERT PAGE
# ===========================================================================
elif page == "📂 Upload & Convert":
    st.header("📂 Upload & Convert")
    st.write(
        "Upload a CSV, Excel (.xlsx), or JSON file. "
        "ByteMind will load it, show a preview, and let you download it "
        "in any other supported format."
    )

    uploaded = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "json"],
        help="Supported formats: CSV, Excel (.xlsx), JSON",
    )

    if uploaded is not None:
        ext = uploaded.name.rsplit(".", 1)[-1].lower()
        st.markdown(f"**Detected file:** `{uploaded.name}` &nbsp;|&nbsp; **Format:** `{ext.upper()}`")

        try:
            df = file_converter.load_file(uploaded, ext)
            # Store in session state so other pages can use it
            st.session_state["dataset"] = df
            st.session_state["dataset_name"] = uploaded.name
        except ValueError as err:
            st.error(f"Could not load the file: {err}")
            st.stop()

        st.divider()
        st.subheader("Dataset Preview")
        col_rows, col_cols = st.columns(2)
        col_rows.metric("Rows", f"{len(df):,}")
        col_cols.metric("Columns", len(df.columns))

        with st.expander("Column names & data types", expanded=False):
            dtype_df = df.dtypes.reset_index()
            dtype_df.columns = ["Column", "Type"]
            st.dataframe(dtype_df, use_container_width=True, hide_index=True)

        st.dataframe(df.head(50), use_container_width=True)
        if len(df) > 50:
            st.caption(f"Showing first 50 of {len(df):,} rows.")

        nested = False
        if ext == "json":
            uploaded.seek(0)
            nested = file_converter.is_nested_json(uploaded)

        st.divider()
        st.subheader("Convert & Download")

        available_formats_clean = []
        for f in ("CSV", "Excel (.xlsx)", "JSON"):
            fmt_ext = "xlsx" if "xlsx" in f.lower() else f.lower()
            if fmt_ext != ext:
                available_formats_clean.append(f)

        target_label = st.selectbox(
            "Target format",
            options=available_formats_clean,
            help="Select the format you want to convert to.",
        )
        _label_to_ext = {"CSV": "csv", "Excel (.xlsx)": "xlsx", "JSON": "json"}
        target_ext = _label_to_ext[target_label]

        conv_warnings = file_converter.get_conversion_warnings(
            df, source_format=ext, target_format=target_ext,
            source_is_nested_json=nested,
        )
        if conv_warnings:
            st.subheader("⚠️ Conversion Warnings")
            for w in conv_warnings:
                st.warning(w)

        try:
            converted_bytes = file_converter.convert_file(df, target_ext)
        except ValueError as err:
            st.error(f"Conversion failed: {err}")
            st.stop()

        st.download_button(
            label=f"⬇️ Download as {target_label}",
            data=converted_bytes,
            file_name=file_converter.DEFAULT_DOWNLOAD_NAMES[target_ext],
            mime=file_converter.MIME_TYPES[target_ext],
        )
    else:
        st.info("Upload a file above to get started.")


# ===========================================================================
# DATA HEALTH PAGE
# ===========================================================================
elif page == "🏥 Data Health":
    st.header("🏥 Data Health Check")
    st.write(
        "Upload your dataset to receive a comprehensive data quality report — "
        "independent of any other feature."
    )

    if not _shared_upload_widget("Data Health"):
        st.stop()

    df = st.session_state["dataset"]
    name = st.session_state["dataset_name"]
    st.caption(f"Analysing: **{name}**")
    st.divider()

    # Run health check
    with st.spinner("Running data health check…"):
        report = data_health.run_health_check(df)

    overview = report["overview"]
    missing = report["missing"]
    duplicates = report["duplicates"]
    dtype_info = report["dtype_info"]
    unique_counts = report["unique_counts"]
    consistency = report["consistency_issues"]
    outliers = report["outliers"]
    score_info = report["health_score"]
    summary = report["quality_summary"]

    # -------------------------------------------------------------------
    # Health Score Banner
    # -------------------------------------------------------------------
    score = score_info["score"]
    grade = score_info["grade"]
    interpretation = score_info["interpretation"]

    score_color = (
        "#2e7d32" if score >= 75
        else "#f57c00" if score >= 45
        else "#c62828"
    )
    st.markdown(
        f"""
        <div style='background:#f7f8fa; border:1px solid #e5e7eb;
                    border-radius:10px; padding:16px 24px; margin-bottom:8px;'>
            <span style='font-size:2rem; font-weight:700; color:{score_color};'>
                {score:.0f} / 100
            </span>
            &nbsp;&nbsp;
            <span style='font-size:1.2rem; font-weight:600; color:{score_color};'>
                Grade {grade}
            </span>
            <br>
            <span style='color:#4a4a6a; font-size:0.9rem;'>{interpretation}</span>
            <br>
            <span style='color:#888; font-size:0.75rem;'>
                ByteMind Data Health Indicator — transparent scoring.
                See "Score Breakdown" below for full details.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------
    # Quality summary cards
    # -------------------------------------------------------------------
    col_good, col_warn, col_attn = st.columns(3)
    with col_good:
        st.markdown("**✅ Good**")
        for item in summary["good"]:
            st.markdown(f"- {item}")
        if not summary["good"]:
            st.markdown("_None_")
    with col_warn:
        st.markdown("**⚠️ Warnings**")
        for item in summary["warnings"]:
            st.markdown(f"- {item}")
        if not summary["warnings"]:
            st.markdown("_None_")
    with col_attn:
        st.markdown("**🔴 Attention**")
        for item in summary["attention"]:
            st.markdown(f"- {item}")
        if not summary["attention"]:
            st.markdown("_None_")

    st.divider()

    # -------------------------------------------------------------------
    # 1. Dataset Overview
    # -------------------------------------------------------------------
    st.subheader("1. Dataset Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Rows", f"{overview['rows']:,}")
    c2.metric("Total Columns", overview["cols"])
    c3.metric("Numeric Cols", len(overview["numeric_cols"]))
    c4.metric("Categorical Cols", len(overview["categorical_cols"]))
    c5.metric("Datetime Cols", len(overview["datetime_cols"]))
    st.caption(f"Memory usage: **{overview['memory_human']}**")

    st.divider()

    # -------------------------------------------------------------------
    # 2. Missing Values
    # -------------------------------------------------------------------
    st.subheader("2. Missing Values")

    import pandas as _pd
    missing_rows = [
        {
            "Column": col,
            "Missing Count": v["count"],
            "Missing %": v["pct"],
            "Substantial": "⚠️ Yes" if v["substantial"] else "✅ No",
        }
        for col, v in missing.items()
    ]
    missing_df = _pd.DataFrame(missing_rows)
    st.dataframe(missing_df, use_container_width=True, hide_index=True)
    total_missing = sum(v["count"] for v in missing.values())
    st.caption(
        f"Total missing cells: {total_missing:,}  |  "
        f"Substantial threshold: ≥{data_health._HIGH_MISSING_PCT}% per column."
    )

    st.divider()

    # -------------------------------------------------------------------
    # 3. Duplicate Rows
    # -------------------------------------------------------------------
    st.subheader("3. Duplicate Rows")
    c_dup1, c_dup2 = st.columns(2)
    c_dup1.metric("Duplicate Rows", f"{duplicates['count']:,}")
    c_dup2.metric("Duplicate %", f"{duplicates['pct']:.2f}%")
    st.caption(
        "Duplicate rows are exact full-row matches. "
        "This is different from repeated values within a single column."
    )

    st.divider()

    # -------------------------------------------------------------------
    # 4. Data Types
    # -------------------------------------------------------------------
    st.subheader("4. Data Types")
    dtype_rows = [
        {
            "Column": d["column"],
            "Pandas Type": d["dtype"],
            "Kind": d["kind"],
            "Possible Date?": "🗓️ Yes" if d["possible_date"] else "—",
        }
        for d in dtype_info
    ]
    st.dataframe(_pd.DataFrame(dtype_rows), use_container_width=True, hide_index=True)
    date_hints = [d["column"] for d in dtype_info if d["possible_date"]]
    if date_hints:
        st.info(
            f"The following column(s) may contain date/time values stored as text: "
            f"**{', '.join(date_hints)}**. "
            "Consider parsing them as datetime if you plan to analyse trends."
        )

    st.divider()

    # -------------------------------------------------------------------
    # 5. Unique Values
    # -------------------------------------------------------------------
    st.subheader("5. Unique Values")
    unique_rows = [
        {
            "Column": col,
            "Unique Values": v["unique"],
            "Unique %": v["pct"],
            "High Cardinality?": "⚠️ Yes" if v["high_cardinality"] else "—",
        }
        for col, v in unique_counts.items()
    ]
    st.dataframe(_pd.DataFrame(unique_rows), use_container_width=True, hide_index=True)
    high_card = [c for c, v in unique_counts.items() if v["high_cardinality"]]
    if high_card:
        st.warning(
            f"High-cardinality columns (≥{int(data_health._HIGH_CARDINALITY_RATIO*100)}% unique): "
            f"**{', '.join(high_card)}**. "
            "These may be identifier columns rather than meaningful categories."
        )

    st.divider()

    # -------------------------------------------------------------------
    # 6. Consistency Issues
    # -------------------------------------------------------------------
    st.subheader("6. Consistency Issues")
    if consistency:
        cons_rows = [
            {"Column": i["column"], "Issue": i["issue"], "Detail": i["detail"], "Count": i["count"]}
            for i in consistency
        ]
        st.dataframe(_pd.DataFrame(cons_rows), use_container_width=True, hide_index=True)
    else:
        st.success("No consistency issues detected.")

    st.divider()

    # -------------------------------------------------------------------
    # 7. Potential Outliers
    # -------------------------------------------------------------------
    st.subheader("7. Potential Outliers (IQR Method)")
    st.caption(
        "Values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR are flagged as *potential* outliers. "
        "This does not mean they are incorrect — domain knowledge is required to judge."
    )

    active_outlier_cols = {c: v for c, v in outliers.items() if not v["skipped"]}
    skipped_outlier_cols = {c: v for c, v in outliers.items() if v["skipped"]}

    if active_outlier_cols:
        outlier_rows = [
            {
                "Column": col,
                "Q1": v["q1"],
                "Q3": v["q3"],
                "IQR": v["iqr"],
                "Lower Fence": v["lower_fence"],
                "Upper Fence": v["upper_fence"],
                "Potential Outliers": v["outlier_count"],
                "Outlier %": v["outlier_pct"],
            }
            for col, v in active_outlier_cols.items()
        ]
        st.dataframe(_pd.DataFrame(outlier_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No numeric columns were suitable for IQR outlier detection.")

    if skipped_outlier_cols:
        with st.expander(f"Skipped columns ({len(skipped_outlier_cols)})", expanded=False):
            for col, v in skipped_outlier_cols.items():
                st.caption(f"**{col}**: {v['skip_reason']}")

    st.divider()

    # -------------------------------------------------------------------
    # 8. Score Breakdown
    # -------------------------------------------------------------------
    with st.expander("📊 Score Breakdown — How the Health Score is Calculated", expanded=False):
        st.markdown(
            """
            The **ByteMind Data Health Score** is a transparency indicator (0–100).
            It is **not** a universal scientific metric.
            Use it as a starting-point guide to data quality, not as a definitive verdict.

            | Factor | Weight | What is measured |
            |---|---|---|
            | Missing values | 30 pts | Fraction of columns with ≥5% missing values |
            | Duplicates | 15 pts | Duplicate-row percentage |
            | Consistency | 25 pts | Fraction of columns with any consistency issue |
            | Outliers | 15 pts | Fraction of numeric columns with ≥5% potential outliers |
            | Cardinality | 15 pts | Fraction of columns flagged as high-cardinality |

            Each factor deducts points proportionally. A perfect 100 means none of
            the above issues were detected.
            """
        )
        components = score_info["components"]
        comp_rows = [
            {
                "Factor": factor,
                "Earned": v["earned"],
                "Max": v["max"],
                "Detail": v["detail"],
            }
            for factor, v in components.items()
        ]
        st.dataframe(_pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)


# ===========================================================================
# ANALYSIS PAGE
# ===========================================================================
elif page == "📊 Analysis":
    st.header("📊 Automatic Data Analysis")
    st.write(
        "Upload a dataset to explore it automatically — descriptive statistics, "
        "correlations, category breakdowns, time trends, and key findings."
    )

    if not _shared_upload_widget("Analysis"):
        st.stop()

    df = st.session_state["dataset"]
    name = st.session_state["dataset_name"]
    st.caption(f"Analysing: **{name}**")

    import pandas as _pd

    ov = analysis.get_overview(df)

    if ov["rows"] == 0:
        st.warning("The loaded dataset is empty. Upload a dataset with rows to run analysis.")
        st.stop()

    # Overview metrics
    st.divider()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{ov['rows']:,}")
    c2.metric("Columns", ov["cols"])
    c3.metric("Numeric", len(ov["numeric_cols"]))
    c4.metric("Categorical", len(ov["categorical_cols"]))
    c5.metric("Datetime", len(ov["datetime_cols"]))
    if ov["total_missing"] > 0:
        st.caption(
            f"ℹ️ {ov['total_missing']:,} missing cells ({ov['total_missing_pct']:.1f}%). "
            "Consider reviewing the Data Health page for details."
        )

    tab_stats, tab_corr, tab_cat, tab_time, tab_topbot, tab_findings = st.tabs([
        "📐 Descriptive Stats",
        "🔗 Correlations",
        "🏷️ Categories",
        "📅 Time Trends",
        "🏆 Top / Bottom",
        "💡 Auto Findings",
    ])

    # -----------------------------------------------------------------------
    # TAB 1 — Descriptive Statistics
    # -----------------------------------------------------------------------
    with tab_stats:
        st.subheader("Descriptive Statistics")
        stats_df = analysis.run_descriptive_stats(df)
        if stats_df.empty:
            st.info("No numeric columns found in this dataset.")
        else:
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            st.caption("NaN and ±inf values are excluded from all calculations.")

            st.divider()
            st.subheader("Column Explorer")
            if ov["numeric_cols"]:
                sel_col = st.selectbox(
                    "Select a numeric column", ov["numeric_cols"], key="stats_col"
                )
                ch_col1, ch_col2 = st.columns(2)
                with ch_col1:
                    st.plotly_chart(
                        analysis.build_histogram(df, sel_col),
                        use_container_width=True,
                    )
                with ch_col2:
                    st.plotly_chart(
                        analysis.build_box_plot(df, sel_col),
                        use_container_width=True,
                    )

    # -----------------------------------------------------------------------
    # TAB 2 — Correlations
    # -----------------------------------------------------------------------
    with tab_corr:
        st.subheader("Pearson Correlation Analysis")
        st.caption(
            "Correlation measures the strength and direction of a linear association "
            "between two numeric variables. **Correlation is not causation.**"
        )
        corr_result = analysis.run_correlation_analysis(df)
        if not corr_result["enough_columns"]:
            st.info(
                "At least 2 numeric columns with variation are needed for correlation analysis."
            )
        else:
            st.plotly_chart(
                analysis.build_correlation_heatmap(corr_result["matrix"]),
                use_container_width=True,
            )
            notable = corr_result["notable_pairs"]
            if notable:
                st.subheader(
                    f"Notable associations (|r| ≥ {analysis.CORRELATION_THRESHOLD})"
                )
                notable_df = _pd.DataFrame([
                    {"Column A": p["col_a"], "Column B": p["col_b"],
                     "r": p["r"], "Direction": p["direction"]}
                    for p in notable
                ])
                st.dataframe(notable_df, use_container_width=True, hide_index=True)
                for p in notable[:5]:
                    st.caption(f"ℹ️ {p['note']}")
            else:
                st.info(
                    f"No pairs exceed the |r| ≥ {analysis.CORRELATION_THRESHOLD} threshold."
                )

    # -----------------------------------------------------------------------
    # TAB 3 — Category Analysis
    # -----------------------------------------------------------------------
    with tab_cat:
        st.subheader("Category / Group Analysis")
        usable_cats = analysis.get_usable_categorical_cols(df)
        if not usable_cats:
            st.info(
                "No suitable categorical columns found. "
                "Columns with too many unique values (possible IDs) are excluded."
            )
        else:
            sel_cat = st.selectbox(
                "Select a categorical column", usable_cats, key="cat_col"
            )
            cat_result = analysis.run_category_analysis(df, sel_cat)
            st.metric(f"Distinct values in '{sel_cat}'", cat_result["total_cats"])
            st.dataframe(
                cat_result["top_n_df"], use_container_width=True, hide_index=True
            )
            st.plotly_chart(
                analysis.build_bar_chart(
                    cat_result["top_n_df"],
                    x_col="value",
                    y_col="count",
                    title=f"Top values in '{sel_cat}'",
                ),
                use_container_width=True,
            )

            if ov["numeric_cols"]:
                st.divider()
                st.subheader("Grouped Summary")
                sel_num = st.selectbox(
                    "Numeric column to summarise", ov["numeric_cols"], key="group_num"
                )
                agg_func = st.radio(
                    "Aggregation", ["mean", "sum", "count"],
                    horizontal=True, key="group_agg"
                )
                try:
                    grouped = analysis.run_grouped_summary(
                        df, cat_col=sel_cat, num_col=sel_num, agg=agg_func
                    )
                    agg_col = f"{agg_func}_{sel_num}"
                    st.dataframe(grouped, use_container_width=True, hide_index=True)
                    st.plotly_chart(
                        analysis.build_bar_chart(
                            grouped,
                            x_col=sel_cat,
                            y_col=agg_col,
                            title=f"{agg_func.capitalize()} of '{sel_num}' by '{sel_cat}'",
                        ),
                        use_container_width=True,
                    )
                except Exception as err:
                    st.error(f"Grouped summary failed: {err}")

    # -----------------------------------------------------------------------
    # TAB 4 — Time Trends
    # -----------------------------------------------------------------------
    with tab_time:
        st.subheader("Date / Time Trend Analysis")
        date_cols = analysis.detect_datetime_columns(df)

        # Offer string columns as potential dates too
        str_cols = [
            c for c in df.columns
            if c not in ov["numeric_cols"] and c not in date_cols
            and not _pd.api.types.is_datetime64_any_dtype(df[c])
        ]
        all_date_candidates = date_cols + str_cols

        if not all_date_candidates:
            st.info("No datetime columns found. Upload a dataset with a date/time column.")
        elif not ov["numeric_cols"]:
            st.info("No numeric columns to plot over time.")
        else:
            sel_date = st.selectbox(
                "Date / time column", all_date_candidates, key="trend_date"
            )
            sel_val = st.selectbox(
                "Numeric measure", ov["numeric_cols"], key="trend_val"
            )
            trend_result = analysis.run_time_series_analysis(df, sel_date, sel_val)
            if not trend_result["enough_data"]:
                st.info(
                    "Not enough data to build a trend. "
                    "Ensure the date column is parseable and there are at least 3 data points."
                )
            else:
                freq_label = "Monthly" if trend_result["freq"] == "ME" else "Daily"
                st.plotly_chart(
                    analysis.build_line_chart(
                        trend_result["trend_df"],
                        title=f"{freq_label} trend of '{sel_val}' over time",
                        x_label="Period",
                        y_label=f"Sum of {sel_val}",
                    ),
                    use_container_width=True,
                )
                st.caption(
                    f"Aggregation: {freq_label.lower()} sum. "
                    f"{len(trend_result['trend_df'])} periods shown."
                )

    # -----------------------------------------------------------------------
    # TAB 5 — Top / Bottom
    # -----------------------------------------------------------------------
    with tab_topbot:
        st.subheader("Top / Bottom Analysis")
        if not ov["numeric_cols"]:
            st.info("No numeric columns available for top/bottom analysis.")
        else:
            sel_val2 = st.selectbox(
                "Numeric column", ov["numeric_cols"], key="topbot_val"
            )
            usable_label_cols = [
                c for c in df.columns if c != sel_val2
            ]
            sel_label = st.selectbox(
                "Label column (optional)",
                ["(none)"] + usable_label_cols,
                key="topbot_label",
            )
            n_topbot = st.slider("N", min_value=3, max_value=20, value=10, key="topbot_n")
            label_col = None if sel_label == "(none)" else sel_label
            try:
                tb = analysis.run_top_bottom(
                    df, value_col=sel_val2, label_col=label_col, n=n_topbot
                )
                col_t, col_b = st.columns(2)
                with col_t:
                    st.markdown(f"**🏆 Top {n_topbot} by '{sel_val2}'**")
                    st.dataframe(tb["top"], use_container_width=True, hide_index=True)
                with col_b:
                    st.markdown(f"**📉 Bottom {n_topbot} by '{sel_val2}'**")
                    st.dataframe(tb["bottom"], use_container_width=True, hide_index=True)
            except Exception as err:
                st.error(f"Top/bottom analysis failed: {err}")

    # -----------------------------------------------------------------------
    # TAB 6 — Automatic Findings
    # -----------------------------------------------------------------------
    with tab_findings:
        st.subheader("💡 Automatic Findings")
        st.caption(
            "These findings are computed directly from the data. "
            "No causal claims are made."
        )
        with st.spinner("Generating findings…"):
            findings = analysis.generate_findings(df)
        if not findings:
            st.info("Not enough data to generate findings.")
        else:
            kind_icon = {
                "quality":     "🔍",
                "numeric":     "📐",
                "correlation": "🔗",
                "category":    "🏷️",
                "trend":       "📅",
            }
            for f in findings:
                icon = kind_icon.get(f["kind"], "•")
                with st.expander(f"{icon} {f['title']}", expanded=True):
                    st.write(f["detail"])


# ===========================================================================
# PLACEHOLDER PAGES
# ===========================================================================
elif page == "🔍 Investigation":
    st.header("🔍 Investigation Engine")
    st.write(
        "ByteMind highlights findings that may deserve further investigation — "
        "based only on what was detected in your data."
    )

    if not _shared_upload_widget("Investigation"):
        st.stop()

    df = st.session_state["dataset"]
    name = st.session_state["dataset_name"]
    st.caption(f"Investigating: **{name}**")
    st.divider()

    import pandas as _pd

    with st.spinner("Generating investigation findings…"):
        findings = investigation.generate_findings(df)

    if not findings:
        st.info(
            "No strong investigation opportunities were detected from the available data. "
            "This may be because the dataset is very small, has only one numeric column, "
            "or the columns do not show notable patterns."
        )
    else:
        st.caption(
            f"Found **{len(findings)}** investigation lead(s). "
            "Expand each card to see the full finding, rationale, and follow-up questions."
        )

        _kind_icon = {
            "correlation":     "🔗",
            "distribution":    "📐",
            "group_variation": "📊",
            "category":        "🏷️",
            "trend":           "📅",
        }

        for i, f in enumerate(findings, 1):
            icon = _kind_icon.get(f["kind"], "🔍")
            with st.expander(f"{icon} {i}. {f['title']}", expanded=(i == 1)):
                st.markdown("**Finding**")
                st.write(f["finding"])

                st.markdown("**Why investigate?**")
                st.write(f["why"])

                st.markdown("**Suggested next step**")
                st.write(f["next_step"])

                if f.get("columns"):
                    st.markdown(
                        "**Relevant columns:** "
                        + ", ".join(f"`{c}`" for c in f["columns"])
                    )

                questions = f.get("questions", [])
                if questions:
                    st.markdown("**Follow-up questions**")
                    for q in questions:
                        st.markdown(f"- {q}")

