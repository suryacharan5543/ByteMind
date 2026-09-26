"""
tests/test_investigation.py
----------------------------
Unit tests for bytemind.investigation.

All tests use small in-memory DataFrames — no external files required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bytemind import investigation


# ---------------------------------------------------------------------------
# Sample DataFrames
# ---------------------------------------------------------------------------

def _correlated_df() -> pd.DataFrame:
    """Two strongly correlated numeric columns."""
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    return pd.DataFrame({
        "x": x,
        "y": [v * 2 + 1 for v in x],   # perfect positive correlation
        "z": [20 - v for v in x],        # perfect negative correlation
    })


def _mixed_df() -> pd.DataFrame:
    """Numeric + categorical + enough rows for group variation."""
    return pd.DataFrame({
        "region":   ["North", "South", "East", "West", "North", "South"] * 3,
        "product":  ["A", "B", "A", "C", "B", "C"] * 3,
        "sales":    [100, 200, 150, 120, 180, 90] * 3,
        "discount": [5.0, 10.0, 7.0, 3.0, 8.0, 6.0] * 3,
    })


def _df_with_dominant_category() -> pd.DataFrame:
    """A categorical column where one value dominates."""
    return pd.DataFrame({
        "status": (["active"] * 18) + (["inactive"] * 2),
        "value":  list(range(20)),
    })


def _df_with_dates() -> pd.DataFrame:
    """DataFrame with a datetime column showing an upward trend."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.DataFrame({
        "date":    dates,
        "revenue": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    })


def _df_high_variability() -> pd.DataFrame:
    """A numeric column with very high coefficient of variation."""
    vals = list(range(1, 11)) + [1000]   # extreme outlier makes CV very high
    return pd.DataFrame({"score": vals, "category": ["A"] * 5 + ["B"] * 6})


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame({"a": pd.Series([], dtype=float)})


def _no_numeric_df() -> pd.DataFrame:
    return pd.DataFrame({"name": ["Alice", "Bob", "Carol"]})


def _single_numeric_df() -> pd.DataFrame:
    """Only one numeric column — correlation impossible."""
    return pd.DataFrame({
        "label": ["x", "y", "z", "x", "y"],
        "value": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


def _constant_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [5.0] * 5, "b": [1.0, 2.0, 3.0, 4.0, 5.0]})


# ===========================================================================
# generate_findings — return shape & required keys
# ===========================================================================

class TestGenerateFindingsShape:
    def test_returns_list(self):
        result = investigation.generate_findings(_mixed_df())
        assert isinstance(result, list)

    def test_each_finding_has_required_keys(self):
        result = investigation.generate_findings(_mixed_df())
        required = {"title", "finding", "why", "next_step", "questions", "columns", "kind"}
        for f in result:
            assert required.issubset(f.keys()), f"Missing keys in finding: {f.keys()}"

    def test_empty_df_returns_empty(self):
        assert investigation.generate_findings(_empty_df()) == []

    def test_no_rows_returns_empty(self):
        df = pd.DataFrame({"a": pd.Series([], dtype=float), "b": pd.Series([], dtype=str)})
        assert investigation.generate_findings(df) == []

    def test_none_raises_value_error(self):
        with pytest.raises(ValueError):
            investigation.generate_findings(None)

    def test_non_dataframe_raises_value_error(self):
        with pytest.raises(ValueError):
            investigation.generate_findings([1, 2, 3])

    def test_max_findings_cap(self):
        """Should never return more than MAX_FINDINGS."""
        result = investigation.generate_findings(_mixed_df())
        assert len(result) <= investigation.MAX_FINDINGS

    def test_titles_are_unique(self):
        result = investigation.generate_findings(_mixed_df())
        titles = [f["title"] for f in result]
        assert len(titles) == len(set(titles))


# ===========================================================================
# Correlation findings
# ===========================================================================

class TestCorrelationFindings:
    def test_strong_positive_correlation_detected(self):
        result = investigation.generate_findings(_correlated_df())
        corr = [f for f in result if f["kind"] == "correlation"]
        assert len(corr) >= 1

    def test_correlation_finding_mentions_association(self):
        result = investigation.generate_findings(_correlated_df())
        corr = [f for f in result if f["kind"] == "correlation"]
        for f in corr:
            assert "association" in f["finding"].lower() or "r =" in f["finding"]

    def test_correlation_finding_no_causation(self):
        """Finding text and 'why' must not claim causation."""
        result = investigation.generate_findings(_correlated_df())
        for f in result:
            for field in ("finding", "why", "next_step"):
                text = f[field].lower()
                assert "caused" not in text
                assert "because of" not in text

    def test_columns_list_contains_both_columns(self):
        result = investigation.generate_findings(_correlated_df())
        corr = [f for f in result if f["kind"] == "correlation"]
        for f in corr:
            assert len(f["columns"]) == 2

    def test_single_column_no_correlation_finding(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        result = investigation.generate_findings(df)
        corr = [f for f in result if f["kind"] == "correlation"]
        assert corr == []

    def test_questions_are_present(self):
        result = investigation.generate_findings(_correlated_df())
        corr = [f for f in result if f["kind"] == "correlation"]
        for f in corr:
            assert isinstance(f["questions"], list)
            assert len(f["questions"]) >= 1

    def test_negative_correlation_detected(self):
        result = investigation.generate_findings(_correlated_df())
        corr = [f for f in result if f["kind"] == "correlation"]
        # x and z should be negative (x goes 1-10, z goes 19 down)
        neg = [f for f in corr if "negative" in f["finding"].lower() or "inverse" in f["finding"].lower()]
        assert len(neg) >= 1


# ===========================================================================
# Distribution / variability findings
# ===========================================================================

class TestDistributionFindings:
    def test_high_variability_detected(self):
        result = investigation.generate_findings(_df_high_variability())
        dist = [f for f in result if f["kind"] == "distribution"]
        assert len(dist) >= 1

    def test_cv_mentioned_in_title(self):
        result = investigation.generate_findings(_df_high_variability())
        dist = [f for f in result if f["kind"] == "distribution"]
        for f in dist:
            assert "cv" in f["title"].lower() or "variab" in f["title"].lower()

    def test_columns_list_contains_affected_column(self):
        result = investigation.generate_findings(_df_high_variability())
        dist = [f for f in result if f["kind"] == "distribution"]
        for f in dist:
            assert len(f["columns"]) >= 1

    def test_low_variability_not_flagged(self):
        """Uniform columns should not trigger a distribution finding."""
        df = pd.DataFrame({"a": [10.0, 10.1, 9.9, 10.0, 10.05]})   # very tight
        result = investigation.generate_findings(df)
        dist = [f for f in result if f["kind"] == "distribution"]
        assert dist == []

    def test_constant_column_not_flagged(self):
        """The constant column 'a' (all 5s, CV=0) must not appear as a distribution
        finding.  Column 'b' has CV≈53% so it may appear — that is correct behaviour."""
        result = investigation.generate_findings(_constant_df())
        dist = [f for f in result if f["kind"] == "distribution"]
        flagged_cols = [col for f in dist for col in f["columns"]]
        assert "a" not in flagged_cols   # constant column must not be flagged


# ===========================================================================
# Group variation findings
# ===========================================================================

class TestGroupVariationFindings:
    def test_group_variation_detected(self):
        # Build a dataset with clear group differences
        df = pd.DataFrame({
            "group":  ["A"] * 10 + ["B"] * 10,
            "value":  [10.0] * 10 + [100.0] * 10,   # 10× ratio
        })
        result = investigation.generate_findings(df)
        gv = [f for f in result if f["kind"] == "group_variation"]
        assert len(gv) >= 1

    def test_group_variation_mentions_ratio(self):
        df = pd.DataFrame({
            "group":  ["A"] * 10 + ["B"] * 10,
            "value":  [10.0] * 10 + [100.0] * 10,
        })
        result = investigation.generate_findings(df)
        gv = [f for f in result if f["kind"] == "group_variation"]
        for f in gv:
            assert "×" in f["title"] or "ratio" in f["finding"].lower()

    def test_no_group_variation_for_similar_groups(self):
        df = pd.DataFrame({
            "group":  ["A", "B", "A", "B", "A", "B"],
            "value":  [10.0, 10.5, 9.8, 10.2, 10.1, 9.9],   # nearly identical
        })
        result = investigation.generate_findings(df)
        gv = [f for f in result if f["kind"] == "group_variation"]
        assert gv == []

    def test_no_categorical_means_no_group_finding(self):
        # Only numeric columns — no categorical to group by
        result = investigation.generate_findings(_correlated_df())
        gv = [f for f in result if f["kind"] == "group_variation"]
        assert gv == []

    def test_group_variation_columns_contain_both(self):
        df = pd.DataFrame({
            "group":  ["A"] * 10 + ["B"] * 10,
            "value":  [10.0] * 10 + [100.0] * 10,
        })
        result = investigation.generate_findings(df)
        gv = [f for f in result if f["kind"] == "group_variation"]
        for f in gv:
            assert len(f["columns"]) == 2


# ===========================================================================
# Category findings
# ===========================================================================

class TestCategoryFindings:
    def test_dominant_category_detected(self):
        result = investigation.generate_findings(_df_with_dominant_category())
        cat = [f for f in result if f["kind"] == "category"]
        assert len(cat) >= 1

    def test_dominant_category_title_mentions_pct(self):
        result = investigation.generate_findings(_df_with_dominant_category())
        cat = [f for f in result if f["kind"] == "category"]
        for f in cat:
            assert "%" in f["title"] or "%" in f["finding"]

    def test_no_category_finding_for_numeric_only(self):
        result = investigation.generate_findings(_correlated_df())
        cat = [f for f in result if f["kind"] == "category"]
        assert cat == []

    def test_top3_concentration_detected(self):
        # 5 categories where top 3 cover 90%
        vals = (["A"] * 45 + ["B"] * 30 + ["C"] * 15 + ["D"] * 5 + ["E"] * 5)
        df = pd.DataFrame({"label": vals, "v": range(100)})
        result = investigation.generate_findings(df)
        cat = [f for f in result if f["kind"] == "category"]
        assert any("top" in f["title"].lower() or "%" in f["title"] for f in cat)


# ===========================================================================
# Trend findings
# ===========================================================================

class TestTrendFindings:
    def test_trend_detected_for_datetime_column(self):
        result = investigation.generate_findings(_df_with_dates())
        trend = [f for f in result if f["kind"] == "trend"]
        assert len(trend) >= 1

    def test_trend_finding_mentions_trend_direction(self):
        result = investigation.generate_findings(_df_with_dates())
        trend = [f for f in result if f["kind"] == "trend"]
        for f in trend:
            assert any(
                word in f["finding"].lower()
                for word in ("upward", "downward", "stable", "trend")
            )

    def test_no_trend_finding_without_datetime(self):
        result = investigation.generate_findings(_mixed_df())
        trend = [f for f in result if f["kind"] == "trend"]
        assert trend == []

    def test_trend_columns_contain_date_and_value(self):
        result = investigation.generate_findings(_df_with_dates())
        trend = [f for f in result if f["kind"] == "trend"]
        for f in trend:
            assert "date" in f["columns"]
            assert len(f["columns"]) == 2


# ===========================================================================
# Follow-up questions
# ===========================================================================

class TestFollowUpQuestions:
    def test_questions_are_non_empty_strings(self):
        result = investigation.generate_findings(_mixed_df())
        for f in result:
            for q in f["questions"]:
                assert isinstance(q, str)
                assert len(q.strip()) > 0

    def test_questions_cap_at_4(self):
        result = investigation.generate_findings(_mixed_df())
        for f in result:
            assert len(f["questions"]) <= 4

    def test_generate_follow_up_questions_accessor(self):
        result = investigation.generate_findings(_mixed_df())
        for f in result:
            qs = investigation.generate_follow_up_questions(f)
            assert isinstance(qs, list)

    def test_questions_reference_column_names(self):
        """Questions should be data-specific, not generic."""
        result = investigation.generate_findings(_correlated_df())
        corr = [f for f in result if f["kind"] == "correlation"]
        for f in corr:
            all_text = " ".join(f["questions"])
            # At least one question should mention one of the columns
            assert any(col in all_text for col in f["columns"])


# ===========================================================================
# Anti-causation enforcement
# ===========================================================================

class TestAntiCausation:
    _forbidden_phrases = ["caused", "because of", "leads to", "results in"]

    def _check_no_causation(self, df: pd.DataFrame) -> None:
        for f in investigation.generate_findings(df):
            for field in ("finding", "why", "next_step"):
                text = f[field].lower()
                for phrase in self._forbidden_phrases:
                    assert phrase not in text, (
                        f"Causation phrase '{phrase}' found in {field}: {text[:200]}"
                    )

    def test_no_causation_correlated_df(self):
        self._check_no_causation(_correlated_df())

    def test_no_causation_mixed_df(self):
        self._check_no_causation(_mixed_df())

    def test_no_causation_dates_df(self):
        self._check_no_causation(_df_with_dates())


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_no_numeric_returns_no_corr_or_dist_findings(self):
        result = investigation.generate_findings(_no_numeric_df())
        for f in result:
            assert f["kind"] not in ("correlation", "distribution", "group_variation")

    def test_constant_numeric_col_no_crash(self):
        result = investigation.generate_findings(_constant_df())
        assert isinstance(result, list)

    def test_single_numeric_col_no_crash(self):
        result = investigation.generate_findings(_single_numeric_df())
        assert isinstance(result, list)

    def test_all_nan_col_no_crash(self):
        df = pd.DataFrame({"a": [float("nan")] * 5, "b": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = investigation.generate_findings(df)
        assert isinstance(result, list)

    def test_inf_values_no_crash(self):
        df = pd.DataFrame({"a": [1.0, float("inf"), 3.0, 4.0, 5.0],
                           "b": [2.0, 4.0, 6.0, 8.0, 10.0]})
        result = investigation.generate_findings(df)
        assert isinstance(result, list)

    def test_single_row_df_no_crash(self):
        df = pd.DataFrame({"x": [42], "y": ["hello"]})
        result = investigation.generate_findings(df)
        assert isinstance(result, list)

    def test_small_dataset_graceful(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = investigation.generate_findings(df)
        assert isinstance(result, list)
