"""
analysis.py
-----------
Automatic Data Analysis (EDA) module for ByteMind.

Responsibilities
----------------
1. Dataset overview      — shape, column breakdown, basic stats
2. Descriptive statistics — per numeric column: count, mean, median, std, min, max, Q1, Q3
3. Correlation analysis   — Pearson matrix, notable pairs above a threshold
4. Category / group analysis — frequency counts, top/bottom categories, grouped summaries
5. Date / time trend analysis — aggregated numeric measure over a datetime column
6. Top / bottom analysis  — top/bottom N rows by a numeric column
7. Automatic findings     — data-derived factual observations (no causation claims)

All functions:
- accept a pandas DataFrame (or simple primitives) and return plain Python dicts/DataFrames
- are side-effect free (never modify the input)
- raise ValueError on invalid input, not NotImplementedError

Chart builders return plotly.graph_objects.Figure objects.
They live here so the Streamlit page stays thin.

No machine learning.  No causation claims.  No hard-coded thresholds presented as scientific fact.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Plotly is only imported at chart-build time so the module can be imported
# and tested without a display environment.
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLOTLY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration constants (documented, not magic)
# ---------------------------------------------------------------------------

#: Minimum Pearson |r| to be considered "notable"
CORRELATION_THRESHOLD: float = 0.5

#: Maximum number of categories shown in bar charts / top-N tables
TOP_N_CATEGORIES: int = 15

#: Maximum unique-value ratio for a column to be treated as categorical (not an ID)
MAX_CATEGORICAL_CARDINALITY_RATIO: float = 0.5

#: Minimum number of rows required to run date-trend analysis
MIN_ROWS_FOR_TREND: int = 3


# ===========================================================================
# 1. Dataset overview
# ===========================================================================

def get_overview(df: pd.DataFrame) -> dict[str, Any]:
    """
    Return structural information about *df*.

    Returns
    -------
    dict with keys:
        rows, cols, numeric_cols (list), categorical_cols (list),
        datetime_cols (list), total_missing (int), total_missing_pct (float)
    """
    _require_df(df)
    numeric = df.select_dtypes(include="number").columns.tolist()
    datetime_ = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric and c not in datetime_]
    total_missing = int(df.isna().sum().sum())
    total_cells = len(df) * len(df.columns)
    return {
        "rows":              len(df),
        "cols":              len(df.columns),
        "numeric_cols":      numeric,
        "categorical_cols":  categorical,
        "datetime_cols":     datetime_,
        "total_missing":     total_missing,
        "total_missing_pct": round(total_missing / total_cells * 100, 2) if total_cells > 0 else 0.0,
    }


# ===========================================================================
# 2. Descriptive statistics
# ===========================================================================

def run_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics for all numeric columns.

    Returns
    -------
    pd.DataFrame with columns:
        column, count, mean, median, std, min, q1, q3, max
    Returns an empty DataFrame if there are no numeric columns.
    """
    _require_df(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return pd.DataFrame(columns=["column", "count", "mean", "median",
                                     "std", "min", "q1", "q3", "max"])
    rows = []
    for col in numeric_cols:
        s = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) == 0:
            rows.append({
                "column": col, "count": 0, "mean": None, "median": None,
                "std": None, "min": None, "q1": None, "q3": None, "max": None,
            })
        else:
            rows.append({
                "column": col,
                "count":  int(len(s)),
                "mean":   round(float(s.mean()), 4),
                "median": round(float(s.median()), 4),
                "std":    round(float(s.std()), 4),
                "min":    round(float(s.min()), 4),
                "q1":     round(float(s.quantile(0.25)), 4),
                "q3":     round(float(s.quantile(0.75)), 4),
                "max":    round(float(s.max()), 4),
            })
    return pd.DataFrame(rows)


# ===========================================================================
# 3. Correlation analysis
# ===========================================================================

def run_correlation_analysis(
    df: pd.DataFrame,
    threshold: float = CORRELATION_THRESHOLD,
) -> dict[str, Any]:
    """
    Pearson correlation matrix and notable pairs.

    Parameters
    ----------
    df : pd.DataFrame
    threshold : float
        Minimum |r| to include a pair in notable_pairs.

    Returns
    -------
    dict with keys:
        matrix          pd.DataFrame  — full correlation matrix (NaN if < 2 numeric cols)
        notable_pairs   list[dict]    — pairs with |r| >= threshold, sorted by |r| desc
        enough_columns  bool          — False when < 2 usable numeric columns
    """
    _require_df(df)
    numeric = (
        df.select_dtypes(include="number")
        .replace([np.inf, -np.inf], np.nan)
    )
    usable = [c for c in numeric.columns if numeric[c].nunique() > 1]

    if len(usable) < 2:
        return {
            "matrix":         pd.DataFrame(),
            "notable_pairs":  [],
            "enough_columns": False,
        }

    matrix = numeric[usable].corr(method="pearson")
    notable: list[dict] = []
    cols = list(matrix.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = matrix.iloc[i, j]
            if pd.isna(r):
                continue
            if abs(r) >= threshold:
                direction = "positive" if r > 0 else "negative"
                notable.append({
                    "col_a":     cols[i],
                    "col_b":     cols[j],
                    "r":         round(float(r), 4),
                    "abs_r":     round(abs(float(r)), 4),
                    "direction": direction,
                    "note":      (
                        f"'{cols[i]}' and '{cols[j]}' show a {direction} association "
                        f"(r = {r:.2f}). This is a correlation, not a cause-effect relationship."
                    ),
                })
    notable.sort(key=lambda x: x["abs_r"], reverse=True)
    return {"matrix": matrix, "notable_pairs": notable, "enough_columns": True}


# ===========================================================================
# 4. Category / group analysis
# ===========================================================================

def get_usable_categorical_cols(
    df: pd.DataFrame,
    max_cardinality_ratio: float = MAX_CATEGORICAL_CARDINALITY_RATIO,
) -> list[str]:
    """
    Return categorical columns that are not likely to be ID columns.

    A column is excluded if unique/rows > max_cardinality_ratio
    or if it has fewer than 2 distinct values (constant).
    """
    _require_df(df)
    total = len(df)
    if total == 0:
        return []
    candidates = df.select_dtypes(exclude="number").columns
    result = []
    for col in candidates:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        n_unique = df[col].nunique()
        if n_unique < 2:
            continue
        if total > 0 and (n_unique / total) > max_cardinality_ratio:
            continue
        result.append(col)
    return result


def run_category_analysis(
    df: pd.DataFrame,
    col: str,
    top_n: int = TOP_N_CATEGORIES,
) -> dict[str, Any]:
    """
    Frequency analysis for a single categorical column.

    Returns
    -------
    dict with keys:
        col          str
        top_n_df     pd.DataFrame  — columns: value, count, pct  (top N by count)
        total_cats   int           — total distinct values
    """
    _require_df(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    counts = (
        df[col].value_counts(dropna=False)
        .head(top_n)
        .reset_index()
    )
    counts.columns = ["value", "count"]
    counts["pct"] = round(counts["count"] / len(df) * 100, 2)
    return {
        "col":        col,
        "top_n_df":   counts,
        "total_cats": int(df[col].nunique(dropna=False)),
    }


def run_grouped_summary(
    df: pd.DataFrame,
    cat_col: str,
    num_col: str,
    agg: str = "mean",
    top_n: int = TOP_N_CATEGORIES,
) -> pd.DataFrame:
    """
    Grouped summary of a numeric column by a categorical column.

    Parameters
    ----------
    agg : str   — 'mean', 'sum', or 'count'

    Returns
    -------
    pd.DataFrame with columns: <cat_col>, <agg>_<num_col>
    """
    _require_df(df)
    for c in (cat_col, num_col):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in DataFrame.")
    allowed = {"mean", "sum", "count"}
    if agg not in allowed:
        raise ValueError(f"agg must be one of {allowed}, got '{agg}'.")

    result = (
        df.groupby(cat_col, observed=True)[num_col]
        .agg(agg)
        .reset_index()
        .rename(columns={num_col: f"{agg}_{num_col}"})
        .sort_values(f"{agg}_{num_col}", ascending=False)
        .head(top_n)
    )
    return result.reset_index(drop=True)


# ===========================================================================
# 5. Date / time trend analysis
# ===========================================================================

def detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    """
    Return columns that are already parsed as datetime dtype.
    Does not attempt to parse string columns here — call try_parse_dates for that.
    """
    _require_df(df)
    return df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()


def try_parse_dates(df: pd.DataFrame, col: str) -> pd.Series | None:
    """
    Attempt to parse *col* as datetime.
    Returns the parsed Series on success, None on failure.
    """
    _require_df(df)
    if col not in df.columns:
        return None
    try:
        parsed = pd.to_datetime(df[col], errors="coerce")
        # Require at least 80 % valid parses
        valid_pct = parsed.notna().mean()
        return parsed if valid_pct >= 0.8 else None
    except Exception:
        return None


def run_time_series_analysis(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
) -> dict[str, Any]:
    """
    Aggregate *value_col* over time using *date_col*.

    Aggregation frequency is chosen automatically:
    - ≥ 90 distinct dates  → monthly
    - otherwise             → daily

    Parameters
    ----------
    date_col  : str  — column to use as the time axis (datetime or parseable)
    value_col : str  — numeric column to aggregate (sum)

    Returns
    -------
    dict with keys:
        trend_df    pd.DataFrame  — columns: period, value
        freq        str           — 'D' or 'ME'  (the pandas freq used)
        enough_data bool
    """
    _require_df(df)
    for c in (date_col, value_col):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found.")

    # Resolve date series
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        date_series = df[date_col]
    else:
        date_series = try_parse_dates(df, date_col)
        if date_series is None:
            return {"trend_df": pd.DataFrame(), "freq": "", "enough_data": False}

    tmp = pd.DataFrame({"_date": date_series, "_val": df[value_col]}).dropna()

    if len(tmp) < MIN_ROWS_FOR_TREND:
        return {"trend_df": pd.DataFrame(), "freq": "", "enough_data": False}

    n_distinct_dates = tmp["_date"].nunique()
    freq = "ME" if n_distinct_dates >= 90 else "D"

    tmp = tmp.set_index("_date").sort_index()
    trend = tmp["_val"].resample(freq).sum().reset_index()
    trend.columns = ["period", "value"]
    trend = trend.dropna()

    return {
        "trend_df":    trend,
        "freq":        freq,
        "enough_data": len(trend) >= 2,
    }


# ===========================================================================
# 6. Top / bottom analysis
# ===========================================================================

def run_top_bottom(
    df: pd.DataFrame,
    value_col: str,
    label_col: str | None = None,
    n: int = 10,
) -> dict[str, Any]:
    """
    Return the top and bottom *n* rows by *value_col*.

    Parameters
    ----------
    label_col : str | None
        If provided, include this column as a label in the results.
        If None, the integer index is used.

    Returns
    -------
    dict with keys:
        top    pd.DataFrame  — top n rows
        bottom pd.DataFrame  — bottom n rows
        col    str
    """
    _require_df(df)
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found.")
    if label_col is not None and label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found.")

    keep_cols = [value_col] if label_col is None else [label_col, value_col]
    valid = df[keep_cols].replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])

    top = valid.nlargest(n, value_col).reset_index(drop=True)
    bottom = valid.nsmallest(n, value_col).reset_index(drop=True)
    return {"top": top, "bottom": bottom, "col": value_col}


# ===========================================================================
# 7. Automatic findings
# ===========================================================================

def generate_findings(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Produce a short list of data-derived factual findings.

    Each finding is a dict:
        title   str  — short headline
        detail  str  — one or two sentences, factual, no causation claims
        kind    str  — 'numeric' | 'correlation' | 'category' | 'trend' | 'quality'

    Returns an empty list if the dataset is too small to yield findings.
    """
    _require_df(df)
    findings: list[dict[str, Any]] = []

    if len(df) == 0 or len(df.columns) == 0:
        return findings

    # --- Quality observation ---
    total_missing = int(df.isna().sum().sum())
    total_cells = len(df) * len(df.columns)
    if total_cells > 0:
        missing_pct = total_missing / total_cells * 100
        if missing_pct > 10:
            findings.append({
                "title":  "Substantial missing data",
                "detail": (
                    f"{total_missing:,} cells ({missing_pct:.1f}%) are missing across the dataset. "
                    "Consider reviewing before drawing conclusions."
                ),
                "kind": "quality",
            })
        elif total_missing == 0:
            findings.append({
                "title":  "No missing values",
                "detail": "The dataset is complete — no missing values were found.",
                "kind":   "quality",
            })

    numeric = (
        df.select_dtypes(include="number")
        .replace([np.inf, -np.inf], np.nan)
    )
    usable_num = [c for c in numeric.columns if numeric[c].nunique() > 1]

    # --- Highest / lowest value observation ---
    for col in usable_num[:3]:        # limit to first 3 to keep findings concise
        s = numeric[col].dropna()
        if len(s) == 0:
            continue
        findings.append({
            "title":  f"Range of '{col}'",
            "detail": (
                f"'{col}' ranges from {s.min():.4g} to {s.max():.4g} "
                f"(mean: {s.mean():.4g}, std: {s.std():.4g})."
            ),
            "kind": "numeric",
        })

    # --- Strongest correlation ---
    if len(usable_num) >= 2:
        corr_result = run_correlation_analysis(df, threshold=0.0)
        if corr_result["notable_pairs"]:
            best = corr_result["notable_pairs"][0]  # already sorted by abs_r desc
            if best["abs_r"] >= CORRELATION_THRESHOLD:
                findings.append({
                    "title":  f"Association: '{best['col_a']}' ↔ '{best['col_b']}'",
                    "detail": best["note"],
                    "kind":   "correlation",
                })

    # --- Top category ---
    cat_cols = get_usable_categorical_cols(df)
    for col in cat_cols[:2]:
        result = run_category_analysis(df, col, top_n=1)
        if len(result["top_n_df"]) > 0:
            top_val = result["top_n_df"].iloc[0]
            findings.append({
                "title":  f"Most common value in '{col}'",
                "detail": (
                    f"'{top_val['value']}' is the most frequent value in '{col}', "
                    f"appearing {int(top_val['count']):,} times "
                    f"({top_val['pct']:.1f}% of rows)."
                ),
                "kind": "category",
            })

    # --- Date trend observation ---
    date_cols = detect_datetime_columns(df)
    if date_cols and usable_num:
        dcol = date_cols[0]
        vcol = usable_num[0]
        trend_result = run_time_series_analysis(df, dcol, vcol)
        if trend_result["enough_data"]:
            tdf = trend_result["trend_df"]
            peak = tdf.loc[tdf["value"].idxmax(), "period"]
            findings.append({
                "title":  f"Trend peak for '{vcol}'",
                "detail": (
                    f"The highest aggregated value of '{vcol}' occurs around "
                    f"{str(peak)[:10]}."
                ),
                "kind": "trend",
            })

    return findings


# ===========================================================================
# Chart builders
# ===========================================================================

def _assert_plotly() -> None:
    if not _PLOTLY_AVAILABLE:                   # pragma: no cover
        raise ImportError("plotly is required to build charts. Install it with: pip install plotly")


def build_histogram(df: pd.DataFrame, col: str) -> "go.Figure":
    """Histogram for a numeric column."""
    _require_df(df)
    _assert_plotly()
    s = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    fig = px.histogram(
        s,
        x=col,
        nbins=min(50, max(10, len(s) // 10)),
        title=f"Distribution of '{col}'",
        labels={col: col, "count": "Count"},
    )
    fig.update_layout(bargap=0.05, xaxis_title=col, yaxis_title="Count")
    return fig


def build_box_plot(df: pd.DataFrame, col: str) -> "go.Figure":
    """Box plot for a numeric column."""
    _require_df(df)
    _assert_plotly()
    s = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    fig = px.box(s, y=col, title=f"Box Plot of '{col}'", labels={col: col})
    fig.update_layout(yaxis_title=col)
    return fig


def build_correlation_heatmap(matrix: pd.DataFrame) -> "go.Figure":
    """Correlation heatmap from a square correlation matrix DataFrame."""
    _assert_plotly()
    if matrix.empty:
        fig = go.Figure()
        fig.update_layout(title="No correlation data available")
        return fig
    fig = px.imshow(
        matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Pearson Correlation Heatmap",
        aspect="auto",
    )
    fig.update_layout(xaxis_title="", yaxis_title="")
    return fig


def build_bar_chart(
    counts_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
) -> "go.Figure":
    """Horizontal bar chart from a frequency / grouped summary DataFrame."""
    _assert_plotly()
    fig = px.bar(
        counts_df,
        x=y_col,
        y=x_col,
        orientation="h",
        title=title,
        labels={x_col: x_col, y_col: y_col},
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        xaxis_title=y_col,
        yaxis_title=x_col,
    )
    return fig


def build_line_chart(
    trend_df: pd.DataFrame,
    title: str,
    x_label: str = "Period",
    y_label: str = "Value",
) -> "go.Figure":
    """Line chart from a two-column DataFrame (period, value)."""
    _assert_plotly()
    fig = px.line(
        trend_df,
        x="period",
        y="value",
        title=title,
        labels={"period": x_label, "value": y_label},
        markers=True,
    )
    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
    return fig


# ===========================================================================
# Internal helpers
# ===========================================================================

def _require_df(df: Any) -> None:
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Expected a non-None pandas DataFrame.")
