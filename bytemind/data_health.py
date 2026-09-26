"""
data_health.py
--------------
Data Health Check module for ByteMind.

Responsibilities
----------------
1. Dataset overview  — shape, column-type breakdown, memory usage
2. Missing values    — per-column count and percentage
3. Duplicate rows    — count and percentage of exact-duplicate rows
4. Data types        — per-column dtype + date-column detection
5. Unique values     — per-column unique count + high-cardinality flagging
6. Consistency checks — empty strings, whitespace, constant columns,
                        mixed types, inf/NaN in numeric columns
7. Outlier detection — IQR-based, per numeric column, labelled "potential"
8. Quality summary   — concise bullet list of positives, warnings, issues
9. Health score      — transparent ByteMind indicator (0–100) with documented
                       factors and weights

All functions accept a pandas DataFrame and return plain Python dicts or
primitive values so they are easy to test without Streamlit.

No data is modified; every function is read-only with respect to the input.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants — health score weights (must sum to 100)
# ---------------------------------------------------------------------------

_SCORE_WEIGHTS: dict[str, int] = {
    "missing":      30,   # penalise substantial missing values
    "duplicates":   15,   # penalise duplicate rows
    "consistency":  25,   # penalise empty strings, constants, mixed types, inf
    "outliers":     15,   # mild penalty for many potential outliers
    "cardinality":  15,   # penalise suspiciously high-cardinality categoricals
}

# Thresholds
_HIGH_MISSING_PCT: float = 5.0     # ≥5 % missing → "substantial"
_CONSTANT_UNIQUE_THRESHOLD: int = 1  # single unique value → constant column
_HIGH_CARDINALITY_RATIO: float = 0.9 # unique/rows ≥ 90 % for categorical col


# ===========================================================================
# 1. Dataset overview
# ===========================================================================

def get_overview(df: pd.DataFrame) -> dict[str, Any]:
    """
    Return basic structural information about *df*.

    Returns
    -------
    dict with keys:
        rows          int
        cols          int
        numeric_cols  list[str]
        categorical_cols list[str]
        datetime_cols list[str]
        memory_bytes  int   — total DataFrame memory usage
        memory_human  str   — human-readable (KB / MB)
    """
    if df is None or df.empty:
        return _empty_overview()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical_cols = [
        c for c in df.columns
        if c not in numeric_cols and c not in datetime_cols
    ]
    mem = int(df.memory_usage(deep=True).sum())

    return {
        "rows":             len(df),
        "cols":             len(df.columns),
        "numeric_cols":     numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols":    datetime_cols,
        "memory_bytes":     mem,
        "memory_human":     _format_bytes(mem),
    }


def _empty_overview() -> dict[str, Any]:
    return {
        "rows": 0, "cols": 0,
        "numeric_cols": [], "categorical_cols": [], "datetime_cols": [],
        "memory_bytes": 0, "memory_human": "0 B",
    }


def _format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 ** 2:.2f} MB"


# ===========================================================================
# 2. Missing values
# ===========================================================================

def get_missing_values(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Return per-column missing-value statistics.

    Returns
    -------
    dict  col -> {'count': int, 'pct': float, 'substantial': bool}
    """
    total = len(df)
    result: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        count = int(df[col].isna().sum())
        pct = round(count / total * 100, 2) if total > 0 else 0.0
        result[col] = {
            "count":       count,
            "pct":         pct,
            "substantial": pct >= _HIGH_MISSING_PCT,
        }
    return result


# ===========================================================================
# 3. Duplicate rows
# ===========================================================================

def get_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    """
    Return duplicate-row statistics.

    Returns
    -------
    dict with keys:
        count  int   — number of duplicate rows (keep='first')
        pct    float — percentage of total rows
    """
    total = len(df)
    count = int(df.duplicated().sum())
    pct = round(count / total * 100, 2) if total > 0 else 0.0
    return {"count": count, "pct": pct}


# ===========================================================================
# 4. Data types + date-column detection
# ===========================================================================

def get_dtype_info(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Return per-column dtype information including a date-detection hint.

    Returns
    -------
    list of dicts, one per column:
        column        str
        dtype         str     — pandas dtype name
        kind          str     — 'numeric' | 'datetime' | 'categorical'
        possible_date bool    — True if non-datetime column may contain dates
    """
    rows = []
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        if pd.api.types.is_numeric_dtype(df[col]):
            kind = "numeric"
            possible_date = False
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            kind = "datetime"
            possible_date = False
        else:
            kind = "categorical"
            possible_date = _looks_like_dates(df[col])
        rows.append({
            "column":        col,
            "dtype":         dtype_str,
            "kind":          kind,
            "possible_date": possible_date,
        })
    return rows


def _looks_like_dates(series: pd.Series, sample: int = 50) -> bool:
    """
    Heuristic: try to parse a sample of non-null string values as dates.
    Returns True if ≥80 % of the sample parses successfully.

    Uses pd.to_datetime without deprecated infer_datetime_format (pandas 2.x).
    """
    sample_vals = series.dropna().head(sample)
    if len(sample_vals) == 0:
        return False
    parsed = 0
    for val in sample_vals:
        try:
            pd.to_datetime(str(val))
            parsed += 1
        except (ValueError, TypeError):
            pass
    return (parsed / len(sample_vals)) >= 0.8


# ===========================================================================
# 5. Unique values
# ===========================================================================

def get_unique_counts(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Return per-column unique-value counts and a high-cardinality flag.

    High cardinality is flagged for non-numeric columns where
    unique / rows ≥ _HIGH_CARDINALITY_RATIO.

    Returns
    -------
    dict  col -> {'unique': int, 'pct': float, 'high_cardinality': bool}
    """
    total = len(df)
    result: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        unique = int(df[col].nunique())
        pct = round(unique / total * 100, 2) if total > 0 else 0.0
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        high_card = (
            not is_numeric
            and total > 0
            and (unique / total) >= _HIGH_CARDINALITY_RATIO
        )
        result[col] = {"unique": unique, "pct": pct, "high_cardinality": high_card}
    return result


# ===========================================================================
# 6. Consistency checks
# ===========================================================================

def get_consistency_issues(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Detect common data quality issues that can be identified without
    knowing the business domain.

    Returns
    -------
    list of issue dicts:
        column   str
        issue    str   — short label
        detail   str   — human-readable explanation
        count    int   — number of affected values (where applicable)
    """
    issues: list[dict[str, Any]] = []

    for col in df.columns:
        series = df[col]

        # Skip columns with no rows — no meaningful checks possible
        if len(series) == 0:
            continue

        # is_string_col covers both object and pandas StringDtype (pandas 2.x)
        is_string_col = (
            series.dtype == object
            or pd.api.types.is_string_dtype(series)
        ) and not pd.api.types.is_numeric_dtype(series)

        # --- empty strings ---
        if is_string_col:
            empty_str = int((series == "").sum())
            if empty_str > 0:
                issues.append({
                    "column": col,
                    "issue":  "Empty strings",
                    "detail": f"{empty_str} cell(s) are empty strings (not NaN).",
                    "count":  empty_str,
                })

        # --- whitespace-only strings ---
        if is_string_col:
            ws_mask = series.apply(
                lambda x: isinstance(x, str) and x.strip() == "" and x != ""
            ).fillna(False)
            ws_count = int(ws_mask.sum())
            if ws_count > 0:
                issues.append({
                    "column": col,
                    "issue":  "Whitespace-only values",
                    "detail": f"{ws_count} cell(s) contain only whitespace.",
                    "count":  ws_count,
                })

        # --- constant column (single unique non-null value) ---
        non_null = series.dropna()
        if len(non_null) > 0 and non_null.nunique() <= _CONSTANT_UNIQUE_THRESHOLD:
            issues.append({
                "column": col,
                "issue":  "Constant column",
                "detail": (
                    f"Column contains only one distinct value "
                    f"('{non_null.iloc[0]}'). May carry no information."
                ),
                "count":  len(non_null),
            })

        # --- mixed types in object columns ---
        if is_string_col:
            type_set = set(series.dropna().apply(type).unique())
            if len(type_set) > 1:
                type_names = sorted(t.__name__ for t in type_set)
                issues.append({
                    "column": col,
                    "issue":  "Mixed data types",
                    "detail": (
                        f"Column contains values of multiple Python types: "
                        f"{', '.join(type_names)}."
                    ),
                    "count":  int(series.notna().sum()),
                })

        # --- inf values in numeric columns ---
        if pd.api.types.is_numeric_dtype(series):
            inf_count = int(np.isinf(series.replace([np.nan], 0)).sum())
            if inf_count > 0:
                issues.append({
                    "column": col,
                    "issue":  "Infinite values",
                    "detail": f"{inf_count} cell(s) contain +inf or -inf.",
                    "count":  inf_count,
                })

    return issues


# ===========================================================================
# 7. Potential outliers (IQR method)
# ===========================================================================

def get_outliers(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    IQR-based potential-outlier detection for numeric columns.

    A value is a *potential* outlier if it falls below Q1 − 1.5×IQR
    or above Q3 + 1.5×IQR.  This does not mean the value is wrong.

    Columns are skipped if they have fewer than 4 non-null values or
    zero IQR (all values identical), as IQR is not meaningful there.

    Returns
    -------
    dict  col -> {
        'q1': float, 'q3': float, 'iqr': float,
        'lower_fence': float, 'upper_fence': float,
        'outlier_count': int, 'outlier_pct': float,
        'skipped': bool, 'skip_reason': str
    }
    """
    result: dict[str, dict[str, Any]] = {}
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        series = df[col].dropna()
        n = len(series)

        if n < 4:
            result[col] = _skipped_outlier("fewer than 4 non-null values")
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1

        if iqr == 0:
            result[col] = _skipped_outlier("IQR is zero (all values identical after dropping NaN)")
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        pct = round(count / n * 100, 2)

        result[col] = {
            "q1":            round(q1, 4),
            "q3":            round(q3, 4),
            "iqr":           round(iqr, 4),
            "lower_fence":   round(lower, 4),
            "upper_fence":   round(upper, 4),
            "outlier_count": count,
            "outlier_pct":   pct,
            "skipped":       False,
            "skip_reason":   "",
        }

    return result


def _skipped_outlier(reason: str) -> dict[str, Any]:
    return {
        "q1": None, "q3": None, "iqr": None,
        "lower_fence": None, "upper_fence": None,
        "outlier_count": 0, "outlier_pct": 0.0,
        "skipped": True, "skip_reason": reason,
    }


# ===========================================================================
# 8. Health score
# ===========================================================================

def compute_health_score(
    missing: dict,
    duplicates: dict,
    consistency_issues: list,
    outliers: dict,
    unique_counts: dict,
    total_rows: int,
    total_cols: int,
) -> dict[str, Any]:
    """
    Compute the ByteMind Data Health Score (0–100).

    This is a *transparency indicator*, not a universal scientific metric.
    Each component is documented below.

    Score factors and weights
    --------------------------
    missing      (30 pts)  — deducted proportionally based on the fraction of
                             columns with ≥5 % missing values.
    duplicates   (15 pts)  — deducted proportionally based on duplicate-row %.
    consistency  (25 pts)  — deducted proportionally based on the fraction of
                             columns that have at least one consistency issue.
    outliers     (15 pts)  — deducted proportionally based on the fraction of
                             numeric columns that have ≥5 % potential outliers.
    cardinality  (15 pts)  — deducted proportionally based on the fraction of
                             non-numeric columns flagged as high-cardinality.

    A perfect 100 means: no missing values, no duplicates, no consistency
    issues, no numeric columns with >5 % potential outliers, and no
    high-cardinality non-numeric columns.

    Returns
    -------
    dict with keys:
        score          float  — 0–100
        grade          str    — A / B / C / D / F
        components     dict   — per-factor (earned, max, detail)
        interpretation str    — plain-language description
    """
    components: dict[str, dict] = {}

    # --- missing ---
    if total_cols > 0:
        substantial_cols = sum(
            1 for v in missing.values() if v["substantial"]
        )
        fraction_missing = substantial_cols / total_cols
    else:
        fraction_missing = 0.0
    missing_score = _SCORE_WEIGHTS["missing"] * (1 - fraction_missing)
    components["missing"] = {
        "earned": round(missing_score, 2),
        "max":    _SCORE_WEIGHTS["missing"],
        "detail": (
            f"{sum(1 for v in missing.values() if v['substantial'])} of "
            f"{total_cols} columns have ≥{_HIGH_MISSING_PCT}% missing values."
        ),
    }

    # --- duplicates ---
    dup_pct = duplicates["pct"] / 100
    dup_score = _SCORE_WEIGHTS["duplicates"] * (1 - min(dup_pct, 1.0))
    components["duplicates"] = {
        "earned": round(dup_score, 2),
        "max":    _SCORE_WEIGHTS["duplicates"],
        "detail": f"{duplicates['count']:,} duplicate rows ({duplicates['pct']:.1f}%).",
    }

    # --- consistency ---
    cols_with_issues = len({i["column"] for i in consistency_issues})
    if total_cols > 0:
        fraction_inconsistent = cols_with_issues / total_cols
    else:
        fraction_inconsistent = 0.0
    consistency_score = _SCORE_WEIGHTS["consistency"] * (1 - fraction_inconsistent)
    components["consistency"] = {
        "earned": round(consistency_score, 2),
        "max":    _SCORE_WEIGHTS["consistency"],
        "detail": (
            f"{cols_with_issues} of {total_cols} columns have consistency issues."
        ),
    }

    # --- outliers ---
    active_outlier_cols = {
        col: v for col, v in outliers.items() if not v["skipped"]
    }
    if active_outlier_cols:
        cols_with_many_outliers = sum(
            1 for v in active_outlier_cols.values() if v["outlier_pct"] >= 5.0
        )
        fraction_outlier = cols_with_many_outliers / len(active_outlier_cols)
    else:
        fraction_outlier = 0.0
    outlier_score = _SCORE_WEIGHTS["outliers"] * (1 - fraction_outlier)
    components["outliers"] = {
        "earned": round(outlier_score, 2),
        "max":    _SCORE_WEIGHTS["outliers"],
        "detail": (
            f"{sum(1 for v in active_outlier_cols.values() if v['outlier_pct'] >= 5.0)} "
            f"numeric columns have ≥5% potential outliers."
        ),
    }

    # --- cardinality ---
    non_numeric_cols = [
        col for col, v in unique_counts.items()
        if not pd.api.types.is_numeric_dtype(pd.Series(dtype="object"))
        # reuse the flag already computed
        if v.get("high_cardinality", False)
    ]
    # Re-derive from unique_counts directly
    high_card_cols = sum(1 for v in unique_counts.values() if v.get("high_cardinality"))
    all_cat_cols = sum(1 for v in unique_counts.values())  # total cols
    if all_cat_cols > 0:
        fraction_high_card = high_card_cols / all_cat_cols
    else:
        fraction_high_card = 0.0
    card_score = _SCORE_WEIGHTS["cardinality"] * (1 - fraction_high_card)
    components["cardinality"] = {
        "earned": round(card_score, 2),
        "max":    _SCORE_WEIGHTS["cardinality"],
        "detail": (
            f"{high_card_cols} non-numeric columns have ≥"
            f"{int(_HIGH_CARDINALITY_RATIO * 100)}% unique values (potential IDs)."
        ),
    }

    total = round(sum(c["earned"] for c in components.values()), 1)
    total = max(0.0, min(100.0, total))

    grade = _letter_grade(total)
    interpretation = _interpret_score(total)

    return {
        "score":          total,
        "grade":          grade,
        "components":     components,
        "interpretation": interpretation,
    }


def _letter_grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    if score >= 45:
        return "D"
    return "F"


def _interpret_score(score: float) -> str:
    if score >= 90:
        return "Excellent — the dataset shows very few quality issues."
    if score >= 75:
        return "Good — minor quality issues worth reviewing before analysis."
    if score >= 60:
        return "Fair — several quality issues detected; review before drawing conclusions."
    if score >= 45:
        return "Poor — significant quality issues; data cleaning recommended."
    return "Very poor — substantial quality problems; thorough cleaning required."


# ===========================================================================
# 9. Quality summary (bullet list)
# ===========================================================================

def get_quality_summary(health_report: dict) -> dict[str, list[str]]:
    """
    Produce a structured plain-language summary from a health report.

    Parameters
    ----------
    health_report : dict
        Output of :func:`run_health_check`.

    Returns
    -------
    dict with keys 'good', 'warnings', 'attention' — each a list[str].
    """
    good: list[str] = []
    warnings: list[str] = []
    attention: list[str] = []

    missing = health_report.get("missing", {})
    dup = health_report.get("duplicates", {})
    consistency = health_report.get("consistency_issues", [])
    outliers = health_report.get("outliers", {})
    unique = health_report.get("unique_counts", {})

    # Missing
    substantial = [c for c, v in missing.items() if v["substantial"]]
    if substantial:
        attention.append(
            f"{len(substantial)} column(s) have ≥{_HIGH_MISSING_PCT}% missing values: "
            f"{', '.join(substantial[:5])}{'...' if len(substantial) > 5 else ''}."
        )
    else:
        good.append("No columns with substantial missing values (≥5%).")

    # Duplicates
    if dup.get("count", 0) > 0:
        warnings.append(
            f"{dup['count']:,} duplicate rows detected ({dup['pct']:.1f}% of total)."
        )
    else:
        good.append("No duplicate rows detected.")

    # Consistency
    issue_types = {}
    for iss in consistency:
        issue_types.setdefault(iss["issue"], 0)
        issue_types[iss["issue"]] += 1
    if issue_types:
        for issue_type, n in sorted(issue_types.items()):
            attention.append(f"Consistency: '{issue_type}' found in {n} column(s).")
    else:
        good.append("No consistency issues detected.")

    # Outliers
    cols_with_outliers = [
        col for col, v in outliers.items()
        if not v.get("skipped") and v.get("outlier_count", 0) > 0
    ]
    if cols_with_outliers:
        warnings.append(
            f"{len(cols_with_outliers)} numeric column(s) contain potential outliers: "
            f"{', '.join(cols_with_outliers[:5])}{'...' if len(cols_with_outliers) > 5 else ''}."
        )
    else:
        numeric_checked = [c for c, v in outliers.items() if not v.get("skipped")]
        if numeric_checked:
            good.append("No potential outliers detected in numeric columns.")

    # High cardinality
    high_card = [c for c, v in unique.items() if v.get("high_cardinality")]
    if high_card:
        warnings.append(
            f"{len(high_card)} column(s) have very high cardinality (possible ID columns): "
            f"{', '.join(high_card[:5])}{'...' if len(high_card) > 5 else ''}."
        )

    return {"good": good, "warnings": warnings, "attention": attention}


# ===========================================================================
# 10. Main entry point
# ===========================================================================

def run_health_check(df: pd.DataFrame) -> dict:
    """
    Run the full data health check and return a single structured report.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict with keys:
        overview, missing, duplicates, dtype_info, unique_counts,
        consistency_issues, outliers, health_score, quality_summary

    Raises
    ------
    ValueError
        If *df* is None or not a DataFrame.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("run_health_check requires a non-None pandas DataFrame.")

    overview = get_overview(df)
    missing = get_missing_values(df)
    duplicates = get_duplicates(df)
    dtype_info = get_dtype_info(df)
    unique_counts = get_unique_counts(df)
    consistency_issues = get_consistency_issues(df)
    outliers = get_outliers(df)

    health_score = compute_health_score(
        missing=missing,
        duplicates=duplicates,
        consistency_issues=consistency_issues,
        outliers=outliers,
        unique_counts=unique_counts,
        total_rows=overview["rows"],
        total_cols=overview["cols"],
    )

    quality_summary = get_quality_summary({
        "missing":            missing,
        "duplicates":         duplicates,
        "consistency_issues": consistency_issues,
        "outliers":           outliers,
        "unique_counts":      unique_counts,
    })

    return {
        "overview":           overview,
        "missing":            missing,
        "duplicates":         duplicates,
        "dtype_info":         dtype_info,
        "unique_counts":      unique_counts,
        "consistency_issues": consistency_issues,
        "outliers":           outliers,
        "health_score":       health_score,
        "quality_summary":    quality_summary,
    }
