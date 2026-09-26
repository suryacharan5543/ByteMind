"""
investigation.py
----------------
Investigation Engine for ByteMind.

Purpose
-------
Analysis tells the user *what* is happening in the data.
The Investigation Engine answers: "What findings deserve a closer look,
why might they matter, and what should I investigate next?"

Design principles
-----------------
- Build on results from bytemind/analysis.py — do not duplicate analysis logic.
- Every finding is grounded in computed data, not generic heuristics.
- No causal claims.  Use language like "associated with", "may indicate",
  "worth investigating", "consider checking".
- No fixed importance numbers assigned without a transparent basis.
- Works with arbitrary column names — does not assume "revenue", "profit", etc.
- Graceful on edge cases (empty df, no numeric cols, no correlations, etc.).

Finding structure
-----------------
Each finding returned is a dict:
    title       str             — short headline
    finding     str             — what was observed (factual)
    why         str             — why it may deserve investigation
    next_step   str             — one concrete suggested next step
    questions   list[str]       — 2–4 follow-up investigation questions
    columns     list[str]       — columns involved
    kind        str             — 'correlation' | 'outlier' | 'category' |
                                  'group_variation' | 'trend' | 'distribution'
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bytemind import analysis as _analysis


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Minimum |r| to generate a correlation finding
STRONG_CORRELATION_THRESHOLD: float = 0.5

#: Minimum |r| to flag a finding as "very strong"
VERY_STRONG_CORRELATION_THRESHOLD: float = 0.8

#: CV (std/mean) threshold above which a column is considered "highly variable"
HIGH_CV_THRESHOLD: float = 0.5

#: Minimum group-mean ratio (max/min) to flag group variation
GROUP_VARIATION_RATIO: float = 2.0

#: Maximum findings returned (keeps UI readable)
MAX_FINDINGS: int = 10


# ===========================================================================
# Public API
# ===========================================================================

def generate_findings(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Produce a list of investigation findings from *df*.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to investigate.

    Returns
    -------
    list[dict]
        Each dict has keys: title, finding, why, next_step, questions,
        columns, kind.
        Returns an empty list if no meaningful findings can be extracted.

    Raises
    ------
    ValueError
        If *df* is None or not a DataFrame.
    """
    _require_df(df)

    if len(df) == 0 or len(df.columns) == 0:
        return []

    findings: list[dict[str, Any]] = []

    findings.extend(_correlation_findings(df))
    findings.extend(_distribution_findings(df))
    findings.extend(_group_variation_findings(df))
    findings.extend(_top_category_findings(df))
    findings.extend(_trend_findings(df))

    # De-duplicate by title and cap
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for f in findings:
        if f["title"] not in seen:
            seen.add(f["title"])
            unique.append(f)

    return unique[:MAX_FINDINGS]


def generate_follow_up_questions(finding: dict[str, Any]) -> list[str]:
    """
    Return the follow-up questions already embedded in *finding*, or an
    empty list if none are present.  Provided as a convenience accessor.
    """
    return finding.get("questions", [])


# ===========================================================================
# Finding detectors
# ===========================================================================

def _correlation_findings(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Generate findings for strong numeric correlations."""
    findings: list[dict[str, Any]] = []

    result = _analysis.run_correlation_analysis(df, threshold=STRONG_CORRELATION_THRESHOLD)
    if not result["enough_columns"]:
        return findings

    cat_cols = _analysis.get_usable_categorical_cols(df)

    for pair in result["notable_pairs"]:
        col_a = pair["col_a"]
        col_b = pair["col_b"]
        r = pair["r"]
        abs_r = pair["abs_r"]
        direction = pair["direction"]

        strength = "very strong" if abs_r >= VERY_STRONG_CORRELATION_THRESHOLD else "strong"
        dir_word = "positive" if direction == "positive" else "negative (inverse)"

        finding_text = (
            f"'{col_a}' and '{col_b}' show a {strength} {dir_word} association "
            f"(Pearson r = {r:.2f})."
        )
        why_text = (
            f"A {strength} association between these two numeric columns may be relevant "
            f"to understanding how they relate in this dataset. "
            f"{'As one increases the other also tends to increase.' if direction == 'positive' else 'As one increases the other tends to decrease.'} "
            f"This is a statistical association — not necessarily a cause-effect relationship."
        )
        next_step = (
            f"Examine a scatter plot of '{col_a}' vs '{col_b}'. "
            f"Check whether the association holds across sub-groups."
        )

        questions = [
            f"Does the association between '{col_a}' and '{col_b}' remain consistent "
            f"across all rows, or is it driven by a subset?",
            f"Are there rows where this pattern breaks down? What do those rows have in common?",
        ]
        if cat_cols:
            questions.append(
                f"Does the relationship between '{col_a}' and '{col_b}' vary "
                f"across different '{cat_cols[0]}' groups?"
            )
        questions.append(
            f"Does the relationship remain after excluding potential outliers?"
        )

        findings.append({
            "title":    f"Association: '{col_a}' ↔ '{col_b}' (r = {r:.2f})",
            "finding":  finding_text,
            "why":      why_text,
            "next_step": next_step,
            "questions": questions[:4],
            "columns":  [col_a, col_b],
            "kind":     "correlation",
        })

    return findings


def _distribution_findings(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Generate findings for highly variable numeric columns."""
    findings: list[dict[str, Any]] = []
    numeric = (
        df.select_dtypes(include="number")
        .replace([np.inf, -np.inf], np.nan)
    )
    cat_cols = _analysis.get_usable_categorical_cols(df)

    for col in numeric.columns:
        s = numeric[col].dropna()
        if len(s) < 4:
            continue
        mean_val = s.mean()
        if mean_val == 0:
            continue
        cv = s.std() / abs(mean_val)
        if cv < HIGH_CV_THRESHOLD:
            continue

        # Check for extreme spread (max/min ratio or large IQR)
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((s < lower) | (s > upper)).sum())
        outlier_pct = round(outlier_count / len(s) * 100, 1)

        finding_text = (
            f"'{col}' shows high variability: mean = {mean_val:.4g}, "
            f"std = {s.std():.4g}, range = [{s.min():.4g}, {s.max():.4g}]. "
            f"Coefficient of variation: {cv:.1%}."
        )
        if outlier_count > 0:
            finding_text += (
                f" {outlier_count} values ({outlier_pct}%) fall outside the IQR fences."
            )

        why_text = (
            f"High variability in '{col}' may indicate meaningful sub-groups, "
            f"data entry issues, or genuinely extreme cases worth understanding. "
        )
        if outlier_count > 0:
            why_text += (
                f"The {outlier_count} potential outlier(s) could be driving summary statistics "
                f"and may deserve individual review."
            )

        next_step = f"Inspect the distribution of '{col}' using a histogram and box plot."

        questions = [
            f"Are the extreme values of '{col}' concentrated in a specific time period or sub-group?",
            f"Do the high/low values of '{col}' correspond to unusual combinations in other columns?",
        ]
        if cat_cols:
            questions.append(
                f"Does the spread of '{col}' differ significantly across '{cat_cols[0]}' groups?"
            )
        if outlier_count > 0:
            questions.append(
                f"Are the {outlier_count} potential outlier(s) in '{col}' legitimate values "
                f"or possible data quality issues?"
            )

        findings.append({
            "title":     f"High variability in '{col}' (CV = {cv:.0%})",
            "finding":   finding_text,
            "why":       why_text,
            "next_step": next_step,
            "questions": questions[:4],
            "columns":   [col],
            "kind":      "distribution",
        })

    return findings


def _group_variation_findings(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Generate findings when the mean of a numeric column varies substantially
    across the values of a categorical column.
    """
    findings: list[dict[str, Any]] = []
    cat_cols = _analysis.get_usable_categorical_cols(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if not cat_cols or not numeric_cols:
        return findings

    for cat_col in cat_cols[:3]:           # cap to keep runtime bounded
        for num_col in numeric_cols[:4]:   # cap to keep runtime bounded
            try:
                grouped = _analysis.run_grouped_summary(df, cat_col, num_col, "mean", top_n=50)
            except Exception:
                continue

            agg_col = f"mean_{num_col}"
            if agg_col not in grouped.columns or len(grouped) < 2:
                continue

            vals = grouped[agg_col].dropna()
            if len(vals) < 2:
                continue

            min_val = vals.min()
            max_val = vals.max()
            if min_val == 0 or max_val == 0:
                continue

            ratio = max_val / min_val if min_val > 0 else abs(max_val / min_val)
            if ratio < GROUP_VARIATION_RATIO:
                continue

            top_cat = grouped.loc[grouped[agg_col].idxmax(), cat_col]
            bot_cat = grouped.loc[grouped[agg_col].idxmin(), cat_col]

            finding_text = (
                f"The mean of '{num_col}' varies substantially across '{cat_col}' groups. "
                f"Highest mean: '{top_cat}' ({max_val:.4g}). "
                f"Lowest mean: '{bot_cat}' ({min_val:.4g}). "
                f"Ratio: {ratio:.1f}×."
            )
            why_text = (
                f"A {ratio:.1f}× difference in average '{num_col}' between '{cat_col}' groups "
                f"may indicate that '{cat_col}' is associated with meaningful differences in '{num_col}'. "
                f"This pattern is worth investigating to understand whether it is consistent, "
                f"and whether other factors might explain it."
            )
            next_step = (
                f"Compare the distribution of '{num_col}' for '{top_cat}' vs '{bot_cat}' "
                f"using box plots or grouped histograms."
            )

            questions = [
                f"Is the difference in '{num_col}' between '{top_cat}' and '{bot_cat}' "
                f"consistent over time, or is it driven by a specific period?",
                f"Are there other variables that differ between '{top_cat}' and '{bot_cat}' "
                f"that might explain the gap in '{num_col}'?",
                f"Is the '{top_cat}' group meaningfully larger in row count, and could that "
                f"affect the comparison?",
                f"Does the pattern hold if potential outliers in '{num_col}' are excluded?",
            ]

            findings.append({
                "title":     f"'{num_col}' varies by '{cat_col}' ({ratio:.1f}× gap)",
                "finding":   finding_text,
                "why":       why_text,
                "next_step": next_step,
                "questions": questions,
                "columns":   [cat_col, num_col],
                "kind":      "group_variation",
            })

    return findings


def _top_category_findings(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Generate findings when a single category dominates a column,
    or when a small number of categories account for most of the data.
    """
    findings: list[dict[str, Any]] = []
    cat_cols = _analysis.get_usable_categorical_cols(df)

    for col in cat_cols[:3]:
        try:
            result = _analysis.run_category_analysis(df, col, top_n=5)
        except Exception:
            continue

        top_df = result["top_n_df"]
        if len(top_df) == 0:
            continue

        top_val = top_df.iloc[0]
        top_pct = float(top_val["pct"])
        total_cats = result["total_cats"]

        # Finding 1: a single value dominates (≥ 50 %)
        if top_pct >= 50.0:
            finding_text = (
                f"'{top_val['value']}' accounts for {top_pct:.1f}% of all rows in '{col}'. "
                f"The column has {total_cats} distinct value(s)."
            )
            why_text = (
                f"When one value dominates a column, analyses that group or filter by '{col}' "
                f"may be heavily influenced by this dominant category. "
                f"It is worth checking whether this concentration is expected or "
                f"may indicate a data collection or scope issue."
            )
            next_step = (
                f"Verify that the dominance of '{top_val['value']}' in '{col}' reflects "
                f"the actual population and is not an artefact of sampling or filtering."
            )
            questions = [
                f"Is '{top_val['value']}' expected to be the most common value in '{col}'?",
                f"Are analyses of other columns affected by this concentration in '{col}'?",
                f"What do the rows with minority '{col}' values look like compared to the majority?",
            ]
            findings.append({
                "title":     f"'{top_val['value']}' dominates '{col}' ({top_pct:.0f}%)",
                "finding":   finding_text,
                "why":       why_text,
                "next_step": next_step,
                "questions": questions,
                "columns":   [col],
                "kind":      "category",
            })

        # Finding 2: top-3 together cover ≥ 80 % with ≥ 5 distinct categories
        elif total_cats >= 5 and len(top_df) >= 3:
            top3_pct = float(top_df.head(3)["pct"].sum())
            if top3_pct >= 80.0:
                top3_vals = ", ".join(f"'{v}'" for v in top_df.head(3)["value"])
                finding_text = (
                    f"The top 3 values ({top3_vals}) account for {top3_pct:.1f}% "
                    f"of all rows in '{col}' (out of {total_cats} distinct values)."
                )
                why_text = (
                    f"High concentration in the top few categories means that the "
                    f"{total_cats - 3} remaining categories together represent only "
                    f"{100 - top3_pct:.1f}% of the data. "
                    f"Analyses involving '{col}' may be dominated by these three groups."
                )
                next_step = (
                    f"Investigate whether the minority categories in '{col}' behave "
                    f"differently from the top 3 on key numeric measures."
                )
                questions = [
                    f"Do the minority '{col}' categories show meaningfully different patterns?",
                    f"Is the long-tail of '{col}' values growing or shrinking over time?",
                    f"Should minority '{col}' categories be grouped for analysis purposes?",
                ]
                findings.append({
                    "title":     f"Top-3 values cover {top3_pct:.0f}% of '{col}'",
                    "finding":   finding_text,
                    "why":       why_text,
                    "next_step": next_step,
                    "questions": questions,
                    "columns":   [col],
                    "kind":      "category",
                })

    return findings


def _trend_findings(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Generate findings from time-series trends if datetime columns exist."""
    findings: list[dict[str, Any]] = []
    date_cols = _analysis.detect_datetime_columns(df)
    if not date_cols:
        return findings

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    usable_num = [
        c for c in numeric_cols
        if df[c].replace([np.inf, -np.inf], np.nan).dropna().nunique() > 1
    ]
    if not usable_num:
        return findings

    date_col = date_cols[0]
    cat_cols = _analysis.get_usable_categorical_cols(df)

    for num_col in usable_num[:2]:      # cap to keep the page readable
        try:
            result = _analysis.run_time_series_analysis(df, date_col, num_col)
        except Exception:
            continue

        if not result["enough_data"]:
            continue

        tdf = result["trend_df"]
        if len(tdf) < 3:
            continue

        values = tdf["value"]
        peak_period = str(tdf.loc[values.idxmax(), "period"])[:10]
        trough_period = str(tdf.loc[values.idxmin(), "period"])[:10]
        peak_val = values.max()
        trough_val = values.min()
        freq_label = "monthly" if result["freq"] == "ME" else "daily"

        # Simple trend direction: compare first and last halves
        half = len(values) // 2
        first_half_mean = values.iloc[:half].mean()
        second_half_mean = values.iloc[half:].mean()
        trend_direction = (
            "upward" if second_half_mean > first_half_mean * 1.05
            else "downward" if second_half_mean < first_half_mean * 0.95
            else "relatively stable"
        )

        finding_text = (
            f"The {freq_label} total of '{num_col}' over '{date_col}' shows a "
            f"{trend_direction} trend. "
            f"Peak: {peak_val:.4g} around {peak_period}. "
            f"Trough: {trough_val:.4g} around {trough_period}."
        )
        why_text = (
            f"A {trend_direction} trend in '{num_col}' over time may indicate a "
            f"meaningful change in the underlying process. "
            f"The difference between peak and trough is "
            f"{abs(peak_val - trough_val):.4g}, which may be worth understanding in context."
        )
        next_step = (
            f"Plot the {freq_label} trend of '{num_col}' and check whether the "
            f"{'peak' if trend_direction == 'upward' else 'trough'} corresponds to a "
            f"known event or seasonal pattern."
        )

        questions = [
            f"Does the {trend_direction} trend in '{num_col}' continue beyond the available data range?",
            f"Are there specific periods where '{num_col}' deviates sharply from the trend?",
        ]
        if cat_cols:
            questions.append(
                f"Is the trend in '{num_col}' consistent across all '{cat_cols[0]}' groups, "
                f"or driven by a specific sub-group?"
            )
        questions.append(
            f"Does the trend change if potential outlier periods are excluded?"
        )

        findings.append({
            "title":     f"'{num_col}' shows {trend_direction} trend over '{date_col}'",
            "finding":   finding_text,
            "why":       why_text,
            "next_step": next_step,
            "questions": questions[:4],
            "columns":   [date_col, num_col],
            "kind":      "trend",
        })

    return findings


# ===========================================================================
# Internal helpers
# ===========================================================================

def _require_df(df: Any) -> None:
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Expected a non-None pandas DataFrame.")
