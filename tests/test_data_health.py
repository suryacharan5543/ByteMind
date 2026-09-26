"""
tests/test_data_health.py
--------------------------
Unit tests for bytemind.data_health.

All tests use small in-memory DataFrames — no external files required.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from bytemind import data_health


# ---------------------------------------------------------------------------
# Sample DataFrames
# ---------------------------------------------------------------------------

def _clean_df() -> pd.DataFrame:
    """A small, clean DataFrame with no quality issues."""
    return pd.DataFrame({
        "name":   ["Alice", "Bob", "Carol", "Dave", "Eve"],
        "age":    [30, 25, 35, 28, 42],
        "city":   ["London", "Paris", "Berlin", "Tokyo", "Sydney"],
        "salary": [55000.0, 48000.0, 62000.0, 51000.0, 70000.0],
    })


def _df_with_missing() -> pd.DataFrame:
    return pd.DataFrame({
        "a": [1, None, 3, None, 5, None, 7, None, 9, None],   # 50% missing
        "b": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],       # 0% missing
        "c": [None, None, None, None, None, 1, 1, 1, 1, 1],   # 50% missing
    })


def _df_with_duplicates() -> pd.DataFrame:
    return pd.DataFrame({
        "x": [1, 2, 3, 1, 2],
        "y": ["a", "b", "c", "a", "b"],
    })


def _df_with_outliers() -> pd.DataFrame:
    """Normal values + extreme outliers in column 'score'."""
    vals = list(range(1, 21))   # 1–20, tightly clustered
    vals.append(1000)           # extreme outlier
    vals.append(-500)           # extreme outlier
    return pd.DataFrame({"score": vals, "category": ["x"] * 22})


def _df_with_consistency_issues() -> pd.DataFrame:
    return pd.DataFrame({
        "name":     ["Alice", "", "  ", "Dave", "Eve"],   # empty + whitespace
        "constant": ["yes", "yes", "yes", "yes", "yes"],  # constant column
        "mixed":    [1, "two", 3, "four", 5],             # mixed types
        "values":   [1.0, 2.0, float("inf"), 4.0, 5.0],  # inf value
    })


def _df_with_dates() -> pd.DataFrame:
    return pd.DataFrame({
        "date_str": ["2024-01-01", "2024-02-15", "2024-03-20"],
        "value":    [100, 200, 300],
    })


def _df_high_cardinality() -> pd.DataFrame:
    """A text column where every value is unique (100% cardinality)."""
    return pd.DataFrame({
        "id":    [f"USR-{i:04d}" for i in range(20)],
        "score": list(range(20)),
    })


# ===========================================================================
# get_overview
# ===========================================================================

class TestGetOverview:
    def test_shape(self):
        df = _clean_df()
        ov = data_health.get_overview(df)
        assert ov["rows"] == 5
        assert ov["cols"] == 4

    def test_numeric_cols_detected(self):
        df = _clean_df()
        ov = data_health.get_overview(df)
        assert set(ov["numeric_cols"]) == {"age", "salary"}

    def test_categorical_cols_detected(self):
        df = _clean_df()
        ov = data_health.get_overview(df)
        assert set(ov["categorical_cols"]) == {"name", "city"}

    def test_datetime_cols_detected(self):
        df = pd.DataFrame({
            "dt": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "val": [1, 2],
        })
        ov = data_health.get_overview(df)
        assert "dt" in ov["datetime_cols"]

    def test_memory_human_not_empty(self):
        ov = data_health.get_overview(_clean_df())
        assert len(ov["memory_human"]) > 0

    def test_empty_df_returns_zeros(self):
        ov = data_health.get_overview(pd.DataFrame())
        assert ov["rows"] == 0
        assert ov["cols"] == 0


# ===========================================================================
# get_missing_values
# ===========================================================================

class TestGetMissingValues:
    def test_no_missing_clean_df(self):
        result = data_health.get_missing_values(_clean_df())
        for v in result.values():
            assert v["count"] == 0
            assert v["pct"] == 0.0
            assert v["substantial"] is False

    def test_counts_correct(self):
        df = _df_with_missing()
        result = data_health.get_missing_values(df)
        assert result["a"]["count"] == 5
        assert result["b"]["count"] == 0
        assert result["c"]["count"] == 5

    def test_percentage_correct(self):
        df = _df_with_missing()
        result = data_health.get_missing_values(df)
        assert result["a"]["pct"] == 50.0

    def test_substantial_flag(self):
        df = _df_with_missing()
        result = data_health.get_missing_values(df)
        assert result["a"]["substantial"] is True   # 50% ≥ 5%
        assert result["b"]["substantial"] is False  # 0%


# ===========================================================================
# get_duplicates
# ===========================================================================

class TestGetDuplicates:
    def test_no_duplicates(self):
        result = data_health.get_duplicates(_clean_df())
        assert result["count"] == 0
        assert result["pct"] == 0.0

    def test_duplicates_detected(self):
        df = _df_with_duplicates()  # rows 0,1 are duplicated by rows 3,4
        result = data_health.get_duplicates(df)
        assert result["count"] == 2
        assert result["pct"] == pytest.approx(40.0, abs=0.01)

    def test_all_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 1]})
        result = data_health.get_duplicates(df)
        assert result["count"] == 2  # first row kept, 2 duplicates


# ===========================================================================
# get_dtype_info
# ===========================================================================

class TestGetDtypeInfo:
    def test_returns_one_row_per_column(self):
        df = _clean_df()
        info = data_health.get_dtype_info(df)
        assert len(info) == len(df.columns)

    def test_numeric_kind(self):
        df = _clean_df()
        info = {d["column"]: d for d in data_health.get_dtype_info(df)}
        assert info["age"]["kind"] == "numeric"
        assert info["salary"]["kind"] == "numeric"

    def test_categorical_kind(self):
        df = _clean_df()
        info = {d["column"]: d for d in data_health.get_dtype_info(df)}
        assert info["name"]["kind"] == "categorical"

    def test_datetime_kind(self):
        df = pd.DataFrame({
            "dt": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        })
        info = data_health.get_dtype_info(df)
        assert info[0]["kind"] == "datetime"
        assert info[0]["possible_date"] is False

    def test_possible_date_detected(self):
        df = _df_with_dates()
        info = {d["column"]: d for d in data_health.get_dtype_info(df)}
        assert info["date_str"]["possible_date"] is True

    def test_numeric_not_flagged_as_date(self):
        df = _clean_df()
        info = {d["column"]: d for d in data_health.get_dtype_info(df)}
        assert info["age"]["possible_date"] is False


# ===========================================================================
# get_unique_counts
# ===========================================================================

class TestGetUniqueCounts:
    def test_correct_counts(self):
        df = _clean_df()
        result = data_health.get_unique_counts(df)
        assert result["name"]["unique"] == 5
        assert result["city"]["unique"] == 5

    def test_high_cardinality_flagged(self):
        df = _df_high_cardinality()
        result = data_health.get_unique_counts(df)
        assert result["id"]["high_cardinality"] is True

    def test_numeric_not_flagged_high_cardinality(self):
        df = _df_high_cardinality()
        result = data_health.get_unique_counts(df)
        assert result["score"]["high_cardinality"] is False

    def test_low_cardinality_not_flagged(self):
        df = pd.DataFrame({"status": ["active", "inactive", "active", "active"]})
        result = data_health.get_unique_counts(df)
        assert result["status"]["high_cardinality"] is False


# ===========================================================================
# get_consistency_issues
# ===========================================================================

class TestGetConsistencyIssues:
    def test_clean_df_no_issues(self):
        issues = data_health.get_consistency_issues(_clean_df())
        assert issues == []

    def test_empty_string_detected(self):
        df = pd.DataFrame({"col": ["Alice", "", "Carol"]})
        issues = data_health.get_consistency_issues(df)
        labels = [i["issue"] for i in issues]
        assert "Empty strings" in labels

    def test_whitespace_only_detected(self):
        df = pd.DataFrame({"col": ["Alice", "  ", "Carol"]})
        issues = data_health.get_consistency_issues(df)
        labels = [i["issue"] for i in issues]
        assert "Whitespace-only values" in labels

    def test_constant_column_detected(self):
        df = pd.DataFrame({"status": ["active"] * 5, "val": [1, 2, 3, 4, 5]})
        issues = data_health.get_consistency_issues(df)
        constant_issues = [i for i in issues if i["issue"] == "Constant column"]
        assert len(constant_issues) == 1
        assert constant_issues[0]["column"] == "status"

    def test_mixed_types_detected(self):
        df = pd.DataFrame({"mixed": [1, "two", 3]})
        issues = data_health.get_consistency_issues(df)
        labels = [i["issue"] for i in issues]
        assert "Mixed data types" in labels

    def test_infinite_values_detected(self):
        df = pd.DataFrame({"val": [1.0, float("inf"), 3.0]})
        issues = data_health.get_consistency_issues(df)
        labels = [i["issue"] for i in issues]
        assert "Infinite values" in labels

    def test_negative_inf_detected(self):
        df = pd.DataFrame({"val": [1.0, float("-inf"), 3.0]})
        issues = data_health.get_consistency_issues(df)
        labels = [i["issue"] for i in issues]
        assert "Infinite values" in labels


# ===========================================================================
# get_outliers
# ===========================================================================

class TestGetOutliers:
    def test_outliers_detected(self):
        df = _df_with_outliers()
        result = data_health.get_outliers(df)
        assert "score" in result
        assert result["score"]["skipped"] is False
        assert result["score"]["outlier_count"] >= 2   # 1000 and -500

    def test_iqr_computed(self):
        df = _df_with_outliers()
        result = data_health.get_outliers(df)
        assert result["score"]["iqr"] > 0

    def test_skipped_when_fewer_than_4_values(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = data_health.get_outliers(df)
        assert result["x"]["skipped"] is True

    def test_skipped_when_zero_iqr(self):
        df = pd.DataFrame({"x": [5.0, 5.0, 5.0, 5.0, 5.0]})
        result = data_health.get_outliers(df)
        assert result["x"]["skipped"] is True

    def test_no_outliers_uniform(self):
        df = pd.DataFrame({"x": list(range(1, 21))})
        result = data_health.get_outliers(df)
        assert result["x"]["skipped"] is False
        assert result["x"]["outlier_count"] == 0

    def test_non_numeric_col_excluded(self):
        df = pd.DataFrame({"text": ["a", "b", "c", "d", "e"]})
        result = data_health.get_outliers(df)
        assert "text" not in result

    def test_outlier_pct_correct(self):
        df = _df_with_outliers()
        result = data_health.get_outliers(df)
        pct = result["score"]["outlier_pct"]
        assert 0.0 < pct <= 100.0


# ===========================================================================
# compute_health_score
# ===========================================================================

class TestComputeHealthScore:
    def _score_for(self, df: pd.DataFrame) -> dict:
        missing = data_health.get_missing_values(df)
        duplicates = data_health.get_duplicates(df)
        consistency = data_health.get_consistency_issues(df)
        outliers = data_health.get_outliers(df)
        unique_counts = data_health.get_unique_counts(df)
        ov = data_health.get_overview(df)
        return data_health.compute_health_score(
            missing, duplicates, consistency, outliers, unique_counts,
            ov["rows"], ov["cols"],
        )

    def test_perfect_score_truly_clean_data(self):
        """A DataFrame with no quality issues AND low-cardinality categoricals
        should score 100. The _clean_df fixture has high-cardinality string
        columns (every name/city is unique), so we use a purpose-built df."""
        df = pd.DataFrame({
            "status":   ["active", "inactive", "active", "inactive", "active"],
            "value":    [10, 20, 30, 40, 50],
            "quantity": [1, 2, 3, 4, 5],
        })
        result = self._score_for(df)
        assert result["score"] == pytest.approx(100.0, abs=0.1)
        assert result["grade"] == "A"

    def test_clean_df_still_scores_high(self):
        """_clean_df has high-cardinality text cols, so score < 100 is expected."""
        result = self._score_for(_clean_df())
        assert result["score"] >= 75.0   # still a good score, grade B or above
        assert result["grade"] in ("A", "B")

    def test_score_in_range(self):
        for df in [_clean_df(), _df_with_missing(), _df_with_duplicates()]:
            result = self._score_for(df)
            assert 0.0 <= result["score"] <= 100.0

    def test_dirty_data_lower_score(self):
        clean = self._score_for(_clean_df())
        dirty = self._score_for(_df_with_consistency_issues())
        assert dirty["score"] < clean["score"]

    def test_components_present(self):
        result = self._score_for(_clean_df())
        assert set(result["components"].keys()) == {
            "missing", "duplicates", "consistency", "outliers", "cardinality"
        }

    def test_grade_mapping(self):
        # Verify letter grades by probing boundary scores
        assert data_health._letter_grade(95) == "A"
        assert data_health._letter_grade(80) == "B"
        assert data_health._letter_grade(65) == "C"
        assert data_health._letter_grade(50) == "D"
        assert data_health._letter_grade(30) == "F"

    def test_interpretation_non_empty(self):
        result = self._score_for(_clean_df())
        assert len(result["interpretation"]) > 0


# ===========================================================================
# get_quality_summary
# ===========================================================================

class TestGetQualitySummary:
    def _summary_for(self, df: pd.DataFrame) -> dict:
        return data_health.get_quality_summary({
            "missing":            data_health.get_missing_values(df),
            "duplicates":         data_health.get_duplicates(df),
            "consistency_issues": data_health.get_consistency_issues(df),
            "outliers":           data_health.get_outliers(df),
            "unique_counts":      data_health.get_unique_counts(df),
        })

    def test_clean_data_has_good_items(self):
        result = self._summary_for(_clean_df())
        assert len(result["good"]) > 0
        assert result["attention"] == []

    def test_missing_values_in_attention(self):
        result = self._summary_for(_df_with_missing())
        assert any("missing" in item.lower() for item in result["attention"])

    def test_duplicates_in_warnings(self):
        result = self._summary_for(_df_with_duplicates())
        assert any("duplicate" in item.lower() for item in result["warnings"])

    def test_outliers_in_warnings(self):
        result = self._summary_for(_df_with_outliers())
        assert any("outlier" in item.lower() for item in result["warnings"])

    def test_returns_correct_keys(self):
        result = self._summary_for(_clean_df())
        assert "good" in result
        assert "warnings" in result
        assert "attention" in result


# ===========================================================================
# run_health_check (integration)
# ===========================================================================

class TestRunHealthCheck:
    def test_returns_all_keys(self):
        report = data_health.run_health_check(_clean_df())
        expected_keys = {
            "overview", "missing", "duplicates", "dtype_info",
            "unique_counts", "consistency_issues", "outliers",
            "health_score", "quality_summary",
        }
        assert set(report.keys()) == expected_keys

    def test_none_raises_value_error(self):
        with pytest.raises(ValueError, match="non-None pandas DataFrame"):
            data_health.run_health_check(None)

    def test_non_dataframe_raises_value_error(self):
        with pytest.raises(ValueError):
            data_health.run_health_check([1, 2, 3])

    def test_clean_df_high_score(self):
        report = data_health.run_health_check(_clean_df())
        assert report["health_score"]["score"] >= 90

    def test_dirty_df_lower_score(self):
        report_clean = data_health.run_health_check(_clean_df())
        report_dirty = data_health.run_health_check(_df_with_consistency_issues())
        assert report_dirty["health_score"]["score"] < report_clean["health_score"]["score"]

    def test_empty_df_runs_without_crash(self):
        """An empty DataFrame should complete without raising an exception."""
        df = pd.DataFrame({"a": pd.Series([], dtype=float), "b": pd.Series([], dtype=str)})
        report = data_health.run_health_check(df)
        assert report["overview"]["rows"] == 0

    def test_single_row_df(self):
        df = pd.DataFrame({"x": [42], "y": ["hello"]})
        report = data_health.run_health_check(df)
        assert report["overview"]["rows"] == 1

    def test_outliers_only_for_numeric(self):
        df = pd.DataFrame({"text": ["a", "b", "c", "d", "e"]})
        report = data_health.run_health_check(df)
        assert report["outliers"] == {}
