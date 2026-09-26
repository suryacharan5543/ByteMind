"""
tests/test_analysis.py
-----------------------
Unit tests for bytemind.analysis.

All tests use small in-memory DataFrames — no external files required.
Chart-builder tests verify that Plotly figures are returned without error;
they do not assert pixel-level rendering.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from bytemind import analysis


# ---------------------------------------------------------------------------
# Sample DataFrames
# ---------------------------------------------------------------------------

def _clean_numeric_df() -> pd.DataFrame:
    """5 rows, 3 numeric columns, no quality issues."""
    return pd.DataFrame({
        "revenue": [100.0, 200.0, 150.0, 300.0, 250.0],
        "cost":    [80.0,  160.0, 120.0, 240.0, 200.0],
        "profit":  [20.0,  40.0,  30.0,  60.0,  50.0],
    })


def _mixed_df() -> pd.DataFrame:
    """Numeric + categorical columns."""
    return pd.DataFrame({
        "region":   ["North", "South", "East", "West", "North", "South"] * 3,
        "product":  ["A", "B", "A", "C", "B", "C"] * 3,
        "sales":    [100, 200, 150, 120, 180, 90] * 3,
        "discount": [5.0, 10.0, 7.0, 3.0, 8.0, 6.0] * 3,
    })


def _df_with_dates() -> pd.DataFrame:
    """DataFrame with a proper datetime column."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.DataFrame({
        "date":    dates,
        "revenue": list(range(10, 110, 10)),
    })


def _df_only_text() -> pd.DataFrame:
    return pd.DataFrame({"name": ["Alice", "Bob", "Carol"]})


def _df_constant_numeric() -> pd.DataFrame:
    return pd.DataFrame({"x": [5.0, 5.0, 5.0, 5.0], "y": [1.0, 2.0, 3.0, 4.0]})


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame({"a": pd.Series([], dtype=float), "b": pd.Series([], dtype=str)})


def _single_row_df() -> pd.DataFrame:
    return pd.DataFrame({"x": [42], "y": ["hello"]})


# ===========================================================================
# get_overview
# ===========================================================================

class TestGetOverview:
    def test_basic_shape(self):
        ov = analysis.get_overview(_mixed_df())
        assert ov["rows"] == 18
        assert ov["cols"] == 4

    def test_numeric_cols(self):
        ov = analysis.get_overview(_mixed_df())
        assert set(ov["numeric_cols"]) == {"sales", "discount"}

    def test_categorical_cols(self):
        ov = analysis.get_overview(_mixed_df())
        assert set(ov["categorical_cols"]) == {"region", "product"}

    def test_datetime_cols(self):
        ov = analysis.get_overview(_df_with_dates())
        assert "date" in ov["datetime_cols"]

    def test_missing_counted(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", None]})
        ov = analysis.get_overview(df)
        assert ov["total_missing"] == 2

    def test_empty_df(self):
        ov = analysis.get_overview(_empty_df())
        assert ov["rows"] == 0

    def test_none_raises(self):
        with pytest.raises(ValueError):
            analysis.get_overview(None)


# ===========================================================================
# run_descriptive_stats
# ===========================================================================

class TestRunDescriptiveStats:
    def test_returns_dataframe(self):
        result = analysis.run_descriptive_stats(_clean_numeric_df())
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self):
        result = analysis.run_descriptive_stats(_clean_numeric_df())
        for col in ("column", "count", "mean", "median", "std", "min", "q1", "q3", "max"):
            assert col in result.columns

    def test_correct_count(self):
        result = analysis.run_descriptive_stats(_clean_numeric_df())
        row = result[result["column"] == "revenue"].iloc[0]
        assert row["count"] == 5

    def test_mean_correct(self):
        result = analysis.run_descriptive_stats(_clean_numeric_df())
        row = result[result["column"] == "revenue"].iloc[0]
        assert row["mean"] == pytest.approx(200.0, abs=0.01)

    def test_min_max_correct(self):
        result = analysis.run_descriptive_stats(_clean_numeric_df())
        row = result[result["column"] == "revenue"].iloc[0]
        assert row["min"] == pytest.approx(100.0)
        assert row["max"] == pytest.approx(300.0)

    def test_no_numeric_returns_empty(self):
        result = analysis.run_descriptive_stats(_df_only_text())
        assert len(result) == 0
        assert "column" in result.columns

    def test_inf_excluded(self):
        df = pd.DataFrame({"x": [1.0, 2.0, float("inf"), 3.0]})
        result = analysis.run_descriptive_stats(df)
        row = result[result["column"] == "x"].iloc[0]
        assert row["count"] == 3
        assert row["max"] == pytest.approx(3.0)

    def test_all_nan_col(self):
        df = pd.DataFrame({"x": [float("nan"), float("nan")]})
        result = analysis.run_descriptive_stats(df)
        row = result[result["column"] == "x"].iloc[0]
        assert row["count"] == 0

    def test_none_raises(self):
        with pytest.raises(ValueError):
            analysis.run_descriptive_stats(None)


# ===========================================================================
# run_correlation_analysis
# ===========================================================================

class TestRunCorrelationAnalysis:
    def test_perfect_positive_correlation(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]})
        result = analysis.run_correlation_analysis(df, threshold=0.9)
        assert result["enough_columns"] is True
        assert len(result["notable_pairs"]) == 1
        assert result["notable_pairs"][0]["r"] == pytest.approx(1.0, abs=0.001)
        assert result["notable_pairs"][0]["direction"] == "positive"

    def test_perfect_negative_correlation(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        result = analysis.run_correlation_analysis(df, threshold=0.9)
        assert result["notable_pairs"][0]["r"] == pytest.approx(-1.0, abs=0.001)
        assert result["notable_pairs"][0]["direction"] == "negative"

    def test_no_correlation_below_threshold(self):
        import random
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"a": rng.normal(0, 1, 50), "b": rng.normal(0, 1, 50)})
        result = analysis.run_correlation_analysis(df, threshold=0.99)
        # Unlikely to have |r| ≥ 0.99 with random data
        for pair in result["notable_pairs"]:
            assert pair["abs_r"] >= 0.99

    def test_single_column_not_enough(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = analysis.run_correlation_analysis(df)
        assert result["enough_columns"] is False
        assert result["notable_pairs"] == []

    def test_constant_column_excluded(self):
        df = _df_constant_numeric()  # x is constant, y varies
        result = analysis.run_correlation_analysis(df)
        assert result["enough_columns"] is False  # x dropped; only y left

    def test_correlation_not_causal(self):
        """The note text must not imply causation."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]})
        result = analysis.run_correlation_analysis(df, threshold=0.0)
        note = result["notable_pairs"][0]["note"]
        assert "not a cause" in note.lower() or "correlation" in note.lower()

    def test_matrix_shape(self):
        df = _clean_numeric_df()
        result = analysis.run_correlation_analysis(df)
        assert result["matrix"].shape == (3, 3)

    def test_sorted_by_abs_r(self):
        df = _clean_numeric_df()
        result = analysis.run_correlation_analysis(df, threshold=0.0)
        pairs = result["notable_pairs"]
        abs_rs = [p["abs_r"] for p in pairs]
        assert abs_rs == sorted(abs_rs, reverse=True)

    def test_none_raises(self):
        with pytest.raises(ValueError):
            analysis.run_correlation_analysis(None)


# ===========================================================================
# get_usable_categorical_cols
# ===========================================================================

class TestGetUsableCategoricalCols:
    def test_returns_categorical(self):
        cols = analysis.get_usable_categorical_cols(_mixed_df())
        assert "region" in cols
        assert "product" in cols

    def test_numeric_excluded(self):
        cols = analysis.get_usable_categorical_cols(_mixed_df())
        assert "sales" not in cols
        assert "discount" not in cols

    def test_high_cardinality_excluded(self):
        df = pd.DataFrame({
            "id":    [f"ID-{i}" for i in range(20)],  # 100% unique
            "label": ["cat1", "cat2"] * 10,
        })
        cols = analysis.get_usable_categorical_cols(df)
        assert "id" not in cols
        assert "label" in cols

    def test_constant_col_excluded(self):
        df = pd.DataFrame({"c": ["x"] * 5, "v": [1, 2, 3, 4, 5]})
        cols = analysis.get_usable_categorical_cols(df)
        assert "c" not in cols

    def test_empty_df_returns_empty(self):
        assert analysis.get_usable_categorical_cols(_empty_df()) == []


# ===========================================================================
# run_category_analysis
# ===========================================================================

class TestRunCategoryAnalysis:
    def test_returns_dict_with_keys(self):
        result = analysis.run_category_analysis(_mixed_df(), "region")
        assert "col" in result
        assert "top_n_df" in result
        assert "total_cats" in result

    def test_top_n_df_columns(self):
        result = analysis.run_category_analysis(_mixed_df(), "region")
        df = result["top_n_df"]
        assert "value" in df.columns
        assert "count" in df.columns
        assert "pct" in df.columns

    def test_counts_sum_less_than_equal_total(self):
        result = analysis.run_category_analysis(_mixed_df(), "region")
        assert result["top_n_df"]["count"].sum() <= len(_mixed_df())

    def test_pct_between_0_and_100(self):
        result = analysis.run_category_analysis(_mixed_df(), "region")
        pcts = result["top_n_df"]["pct"]
        assert (pcts >= 0).all() and (pcts <= 100).all()

    def test_top_n_respected(self):
        result = analysis.run_category_analysis(_mixed_df(), "region", top_n=1)
        assert len(result["top_n_df"]) == 1

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="not found"):
            analysis.run_category_analysis(_mixed_df(), "nonexistent")


# ===========================================================================
# run_grouped_summary
# ===========================================================================

class TestRunGroupedSummary:
    def test_mean_grouping(self):
        result = analysis.run_grouped_summary(_mixed_df(), "region", "sales", "mean")
        assert "mean_sales" in result.columns
        assert "region" in result.columns

    def test_sum_grouping(self):
        result = analysis.run_grouped_summary(_mixed_df(), "region", "sales", "sum")
        assert "sum_sales" in result.columns

    def test_count_grouping(self):
        result = analysis.run_grouped_summary(_mixed_df(), "region", "sales", "count")
        assert "count_sales" in result.columns

    def test_invalid_agg_raises(self):
        with pytest.raises(ValueError, match="agg must be one of"):
            analysis.run_grouped_summary(_mixed_df(), "region", "sales", "median")

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="not found"):
            analysis.run_grouped_summary(_mixed_df(), "region", "nonexistent")

    def test_sorted_descending(self):
        result = analysis.run_grouped_summary(_mixed_df(), "region", "sales", "sum")
        col = "sum_sales"
        assert list(result[col]) == sorted(result[col], reverse=True)


# ===========================================================================
# detect_datetime_columns / try_parse_dates
# ===========================================================================

class TestDatetimeFunctions:
    def test_detect_datetime_cols(self):
        df = _df_with_dates()
        cols = analysis.detect_datetime_columns(df)
        assert "date" in cols

    def test_no_datetime_cols(self):
        assert analysis.detect_datetime_columns(_clean_numeric_df()) == []

    def test_try_parse_dates_success(self):
        df = pd.DataFrame({"d": ["2024-01-01", "2024-02-01", "2024-03-01"]})
        parsed = analysis.try_parse_dates(df, "d")
        assert parsed is not None
        assert pd.api.types.is_datetime64_any_dtype(parsed)

    def test_try_parse_dates_failure(self):
        df = pd.DataFrame({"d": ["not_a_date", "also_not", "nope"]})
        assert analysis.try_parse_dates(df, "d") is None

    def test_try_parse_dates_missing_col(self):
        assert analysis.try_parse_dates(_clean_numeric_df(), "nonexistent") is None


# ===========================================================================
# run_time_series_analysis
# ===========================================================================

class TestRunTimeSeriesAnalysis:
    def test_returns_trend_df(self):
        df = _df_with_dates()
        result = analysis.run_time_series_analysis(df, "date", "revenue")
        assert result["enough_data"] is True
        assert "period" in result["trend_df"].columns
        assert "value" in result["trend_df"].columns

    def test_daily_freq_for_small_dataset(self):
        df = _df_with_dates()
        result = analysis.run_time_series_analysis(df, "date", "revenue")
        assert result["freq"] == "D"

    def test_monthly_freq_for_large_dataset(self):
        dates = pd.date_range("2020-01-01", periods=120, freq="D")
        df = pd.DataFrame({"date": dates, "value": range(120)})
        result = analysis.run_time_series_analysis(df, "date", "value")
        assert result["freq"] == "ME"

    def test_string_date_col_parsed(self):
        df = pd.DataFrame({
            "date":  ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "sales": [10, 20, 30, 40],
        })
        result = analysis.run_time_series_analysis(df, "date", "sales")
        assert result["enough_data"] is True

    def test_unparseable_date_returns_no_data(self):
        df = pd.DataFrame({
            "date":  ["not-a-date"] * 5,
            "sales": [1, 2, 3, 4, 5],
        })
        result = analysis.run_time_series_analysis(df, "date", "sales")
        assert result["enough_data"] is False

    def test_too_few_rows_returns_no_data(self):
        df = pd.DataFrame({
            "date":  pd.date_range("2024-01-01", periods=2),
            "sales": [1, 2],
        })
        result = analysis.run_time_series_analysis(df, "date", "sales")
        assert result["enough_data"] is False

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="not found"):
            analysis.run_time_series_analysis(_df_with_dates(), "date", "nonexistent")


# ===========================================================================
# run_top_bottom
# ===========================================================================

class TestRunTopBottom:
    def test_top_n_correct(self):
        df = _clean_numeric_df()
        result = analysis.run_top_bottom(df, "revenue", n=3)
        assert len(result["top"]) == 3

    def test_bottom_n_correct(self):
        df = _clean_numeric_df()
        result = analysis.run_top_bottom(df, "revenue", n=3)
        assert len(result["bottom"]) == 3

    def test_top_values_descending(self):
        df = _clean_numeric_df()
        result = analysis.run_top_bottom(df, "revenue", n=5)
        vals = result["top"]["revenue"].tolist()
        assert vals == sorted(vals, reverse=True)

    def test_bottom_values_ascending(self):
        df = _clean_numeric_df()
        result = analysis.run_top_bottom(df, "revenue", n=5)
        vals = result["bottom"]["revenue"].tolist()
        assert vals == sorted(vals)

    def test_with_label_col(self):
        df = _mixed_df()
        result = analysis.run_top_bottom(df, "sales", label_col="region", n=5)
        assert "region" in result["top"].columns
        assert "sales" in result["top"].columns

    def test_n_larger_than_df_returns_all(self):
        df = _clean_numeric_df()  # 5 rows
        result = analysis.run_top_bottom(df, "revenue", n=100)
        assert len(result["top"]) == 5

    def test_missing_value_col_raises(self):
        with pytest.raises(ValueError, match="not found"):
            analysis.run_top_bottom(_clean_numeric_df(), "nonexistent")

    def test_missing_label_col_raises(self):
        with pytest.raises(ValueError, match="Label column"):
            analysis.run_top_bottom(_clean_numeric_df(), "revenue", label_col="bad")

    def test_inf_excluded(self):
        df = pd.DataFrame({"x": [1.0, float("inf"), 3.0, 4.0, 5.0]})
        result = analysis.run_top_bottom(df, "x", n=10)
        assert not any(np.isinf(v) for v in result["top"]["x"])


# ===========================================================================
# generate_findings
# ===========================================================================

class TestGenerateFindings:
    def test_returns_list(self):
        findings = analysis.generate_findings(_clean_numeric_df())
        assert isinstance(findings, list)

    def test_each_finding_has_required_keys(self):
        findings = analysis.generate_findings(_mixed_df())
        for f in findings:
            assert "title" in f
            assert "detail" in f
            assert "kind" in f

    def test_no_causation_in_detail(self):
        """Detail text must not use 'caused' or 'because of'."""
        findings = analysis.generate_findings(_mixed_df())
        for f in findings:
            detail_lower = f["detail"].lower()
            assert "caused" not in detail_lower
            assert "because of" not in detail_lower

    def test_empty_df_returns_empty(self):
        assert analysis.generate_findings(_empty_df()) == []

    def test_findings_for_clean_numeric(self):
        findings = analysis.generate_findings(_clean_numeric_df())
        kinds = {f["kind"] for f in findings}
        # Should have at least numeric range findings + correlation
        assert "numeric" in kinds or "correlation" in kinds

    def test_category_finding_present(self):
        findings = analysis.generate_findings(_mixed_df())
        kinds = {f["kind"] for f in findings}
        assert "category" in kinds

    def test_missing_data_finding(self):
        df = pd.DataFrame({
            "a": [None] * 50 + [1] * 50,
            "b": [None] * 40 + [2] * 60,
        })
        findings = analysis.generate_findings(df)
        quality = [f for f in findings if f["kind"] == "quality"]
        assert len(quality) > 0

    def test_none_raises(self):
        with pytest.raises(ValueError):
            analysis.generate_findings(None)


# ===========================================================================
# Chart builders (smoke tests — verify Figures returned without crash)
# ===========================================================================

class TestChartBuilders:
    def test_histogram_returns_figure(self):
        fig = analysis.build_histogram(_clean_numeric_df(), "revenue")
        assert fig is not None
        assert hasattr(fig, "data")

    def test_box_plot_returns_figure(self):
        fig = analysis.build_box_plot(_clean_numeric_df(), "revenue")
        assert fig is not None

    def test_correlation_heatmap_non_empty_matrix(self):
        result = analysis.run_correlation_analysis(_clean_numeric_df())
        fig = analysis.build_correlation_heatmap(result["matrix"])
        assert fig is not None

    def test_correlation_heatmap_empty_matrix(self):
        fig = analysis.build_correlation_heatmap(pd.DataFrame())
        assert fig is not None

    def test_bar_chart_returns_figure(self):
        cat_result = analysis.run_category_analysis(_mixed_df(), "region")
        fig = analysis.build_bar_chart(
            cat_result["top_n_df"], x_col="value", y_col="count",
            title="Test"
        )
        assert fig is not None

    def test_line_chart_returns_figure(self):
        result = analysis.run_time_series_analysis(_df_with_dates(), "date", "revenue")
        fig = analysis.build_line_chart(result["trend_df"], title="Test")
        assert fig is not None


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_single_row_df(self):
        df = _single_row_df()
        # Should not crash; findings may be empty
        findings = analysis.generate_findings(df)
        assert isinstance(findings, list)

    def test_all_nan_numeric_col(self):
        df = pd.DataFrame({"x": [float("nan")] * 5})
        stats = analysis.run_descriptive_stats(df)
        assert stats.iloc[0]["count"] == 0

    def test_corr_with_one_non_constant_col(self):
        # Only one non-constant numeric col → not enough for correlation
        df = pd.DataFrame({"x": [1, 1, 1, 1], "y": [1, 2, 3, 4]})
        result = analysis.run_correlation_analysis(df)
        assert result["enough_columns"] is False

    def test_top_bottom_empty_after_nan_drop(self):
        df = pd.DataFrame({"x": [float("nan"), float("nan")]})
        result = analysis.run_top_bottom(df, "x", n=5)
        assert len(result["top"]) == 0
        assert len(result["bottom"]) == 0

    def test_grouped_summary_empty_df(self):
        df = pd.DataFrame({"cat": pd.Series([], dtype=str), "val": pd.Series([], dtype=float)})
        result = analysis.run_grouped_summary(df, "cat", "val", "mean")
        assert isinstance(result, pd.DataFrame)
