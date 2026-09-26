"""
tests/test_file_converter.py
----------------------------
Unit tests for bytemind.file_converter.

All tests use in-memory data — no files are written to disk.
"""

from __future__ import annotations

import io
import json

import pandas as pd
import pytest

from bytemind import file_converter


# ---------------------------------------------------------------------------
# Helpers — in-memory file factories
# ---------------------------------------------------------------------------

def _make_csv(content: str) -> io.BytesIO:
    return io.BytesIO(content.encode("utf-8"))


def _make_json(data) -> io.BytesIO:
    return io.BytesIO(json.dumps(data).encode("utf-8"))


def _make_excel(df: pd.DataFrame) -> io.BytesIO:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_CSV = "name,age,city\nAlice,30,London\nBob,25,Paris\nCarol,35,Berlin\n"

SAMPLE_RECORDS = [
    {"name": "Alice", "age": 30, "city": "London"},
    {"name": "Bob",   "age": 25, "city": "Paris"},
    {"name": "Carol", "age": 35, "city": "Berlin"},
]

NESTED_JSON = [
    {"name": "Alice", "address": {"city": "London", "zip": "E1"}},
    {"name": "Bob",   "address": {"city": "Paris",  "zip": "75001"}},
]


# ===========================================================================
# load_file — CSV
# ===========================================================================

class TestLoadCSV:
    def test_basic_load(self):
        df = file_converter.load_file(_make_csv(SAMPLE_CSV), "csv")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["name", "age", "city"]
        assert len(df) == 3

    def test_values_correct(self):
        df = file_converter.load_file(_make_csv(SAMPLE_CSV), "csv")
        assert df.iloc[0]["name"] == "Alice"
        assert df.iloc[1]["age"] == 25

    def test_extension_with_dot_prefix(self):
        """Extension may arrive with a leading dot — should still work."""
        df = file_converter.load_file(_make_csv(SAMPLE_CSV), ".csv")
        assert len(df) == 3

    def test_latin1_encoding(self):
        """CSV with latin-1 content (e.g. accented characters) loads correctly."""
        content = "name,city\nAntoiné,Montréal\n".encode("latin-1")
        df = file_converter.load_file(io.BytesIO(content), "csv")
        assert "name" in df.columns

    def test_malformed_csv_raises_value_error(self):
        """A CSV with an unterminated quote (ParserError) raises ValueError."""
        # An unterminated quote triggers pd.errors.ParserError on every encoding.
        bad = io.BytesIO(b"name,value\n\"unterminated,1\n")
        with pytest.raises(ValueError, match="Could not read the CSV file"):
            file_converter.load_file(bad, "csv")


# ===========================================================================
# load_file — Excel
# ===========================================================================

class TestLoadExcel:
    def test_basic_load(self):
        source_df = pd.DataFrame(SAMPLE_RECORDS)
        xl = _make_excel(source_df)
        df = file_converter.load_file(xl, "xlsx")
        assert list(df.columns) == ["name", "age", "city"]
        assert len(df) == 3

    def test_values_correct(self):
        source_df = pd.DataFrame(SAMPLE_RECORDS)
        xl = _make_excel(source_df)
        df = file_converter.load_file(xl, "xlsx")
        assert df.iloc[2]["city"] == "Berlin"

    def test_invalid_bytes_raises_value_error(self):
        bad = io.BytesIO(b"this is not an xlsx file")
        with pytest.raises(ValueError, match="Could not read the Excel file"):
            file_converter.load_file(bad, "xlsx")


# ===========================================================================
# load_file — JSON
# ===========================================================================

class TestLoadJSON:
    def test_list_of_records(self):
        df = file_converter.load_file(_make_json(SAMPLE_RECORDS), "json")
        assert list(df.columns) == ["name", "age", "city"]
        assert len(df) == 3

    def test_nested_json_flattened(self):
        """Nested JSON should load and produce dot-notation columns."""
        df = file_converter.load_file(_make_json(NESTED_JSON), "json")
        assert "name" in df.columns
        # Flattened nested key
        assert any("address" in c for c in df.columns)

    def test_invalid_json_raises_value_error(self):
        bad = io.BytesIO(b"{ this is not json }")
        with pytest.raises(ValueError, match="Invalid JSON"):
            file_converter.load_file(bad, "json")

    def test_empty_list_raises_value_error(self):
        empty = io.BytesIO(b"[]")
        with pytest.raises(ValueError, match="empty table"):
            file_converter.load_file(empty, "json")


# ===========================================================================
# load_file — unsupported format
# ===========================================================================

class TestLoadUnsupportedFormat:
    def test_unsupported_extension_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported file format"):
            file_converter.load_file(io.BytesIO(b""), "parquet")

    def test_xml_extension_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported file format"):
            file_converter.load_file(io.BytesIO(b"<root/>"), "xml")


# ===========================================================================
# is_nested_json
# ===========================================================================

class TestIsNestedJSON:
    def test_flat_list_not_nested(self):
        assert file_converter.is_nested_json(_make_json(SAMPLE_RECORDS)) is False

    def test_nested_list_is_nested(self):
        assert file_converter.is_nested_json(_make_json(NESTED_JSON)) is True

    def test_deeply_nested_is_nested(self):
        deep = [{"a": {"b": {"c": 1}}}]
        assert file_converter.is_nested_json(_make_json(deep)) is True

    def test_invalid_json_returns_false(self):
        assert file_converter.is_nested_json(io.BytesIO(b"not json")) is False


# ===========================================================================
# convert_file — CSV
# ===========================================================================

class TestConvertToCSV:
    def test_produces_bytes(self):
        df = pd.DataFrame(SAMPLE_RECORDS)
        result = file_converter.convert_file(df, "csv")
        assert isinstance(result, bytes)

    def test_csv_roundtrip(self):
        df = pd.DataFrame(SAMPLE_RECORDS)
        csv_bytes = file_converter.convert_file(df, "csv")
        reloaded = pd.read_csv(io.BytesIO(csv_bytes))
        assert list(reloaded.columns) == ["name", "age", "city"]
        assert len(reloaded) == 3

    def test_csv_no_index(self):
        df = pd.DataFrame(SAMPLE_RECORDS)
        csv_bytes = file_converter.convert_file(df, "csv")
        text = csv_bytes.decode("utf-8")
        # No unnamed index column in header
        assert text.startswith("name,age,city")


# ===========================================================================
# convert_file — Excel
# ===========================================================================

class TestConvertToExcel:
    def test_produces_bytes(self):
        df = pd.DataFrame(SAMPLE_RECORDS)
        result = file_converter.convert_file(df, "xlsx")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_excel_roundtrip(self):
        df = pd.DataFrame(SAMPLE_RECORDS)
        xlsx_bytes = file_converter.convert_file(df, "xlsx")
        reloaded = pd.read_excel(io.BytesIO(xlsx_bytes), engine="openpyxl")
        assert list(reloaded.columns) == ["name", "age", "city"]
        assert len(reloaded) == 3


# ===========================================================================
# convert_file — JSON
# ===========================================================================

class TestConvertToJSON:
    def test_produces_bytes(self):
        df = pd.DataFrame(SAMPLE_RECORDS)
        result = file_converter.convert_file(df, "json")
        assert isinstance(result, bytes)

    def test_json_roundtrip(self):
        df = pd.DataFrame(SAMPLE_RECORDS)
        json_bytes = file_converter.convert_file(df, "json")
        parsed = json.loads(json_bytes.decode("utf-8"))
        assert isinstance(parsed, list)
        assert len(parsed) == 3
        assert parsed[0]["name"] == "Alice"

    def test_json_orient_records(self):
        df = pd.DataFrame(SAMPLE_RECORDS)
        json_bytes = file_converter.convert_file(df, "json")
        parsed = json.loads(json_bytes.decode("utf-8"))
        # orient=records → list of dicts, not dict-of-lists
        assert isinstance(parsed[0], dict)


# ===========================================================================
# convert_file — unsupported format
# ===========================================================================

class TestConvertUnsupportedFormat:
    def test_parquet_raises_value_error(self):
        df = pd.DataFrame(SAMPLE_RECORDS)
        with pytest.raises(ValueError, match="Unsupported target format"):
            file_converter.convert_file(df, "parquet")


# ===========================================================================
# CSV → JSON  (end-to-end round-trip through both functions)
# ===========================================================================

class TestCSVToJSONEndToEnd:
    def test_csv_to_json(self):
        df = file_converter.load_file(_make_csv(SAMPLE_CSV), "csv")
        json_bytes = file_converter.convert_file(df, "json")
        parsed = json.loads(json_bytes.decode("utf-8"))
        assert len(parsed) == 3
        cities = {r["city"] for r in parsed}
        assert cities == {"London", "Paris", "Berlin"}


# ===========================================================================
# CSV → Excel  (end-to-end round-trip through both functions)
# ===========================================================================

class TestCSVToExcelEndToEnd:
    def test_csv_to_excel(self):
        df = file_converter.load_file(_make_csv(SAMPLE_CSV), "csv")
        xlsx_bytes = file_converter.convert_file(df, "xlsx")
        reloaded = pd.read_excel(io.BytesIO(xlsx_bytes), engine="openpyxl")
        assert len(reloaded) == 3
        assert "name" in reloaded.columns


# ===========================================================================
# get_conversion_warnings
# ===========================================================================

class TestConversionWarnings:
    def test_no_warnings_for_clean_csv_to_json(self):
        df = pd.DataFrame(SAMPLE_RECORDS)
        warnings = file_converter.get_conversion_warnings(df, "csv", "json")
        assert warnings == []

    def test_nested_json_warning_appears(self):
        df = file_converter.load_file(_make_json(NESTED_JSON), "json")
        warnings = file_converter.get_conversion_warnings(
            df, "json", "csv", source_is_nested_json=True
        )
        assert any("nested" in w.lower() for w in warnings)

    def test_no_nested_warning_when_flag_false(self):
        df = file_converter.load_file(_make_json(NESTED_JSON), "json")
        warnings = file_converter.get_conversion_warnings(
            df, "json", "csv", source_is_nested_json=False
        )
        assert not any("nested" in w.lower() for w in warnings)

    def test_comma_in_column_name_warns_for_csv(self):
        df = pd.DataFrame({"col,name": [1, 2], "value": [3, 4]})
        warnings = file_converter.get_conversion_warnings(df, "json", "csv")
        assert any("comma" in w.lower() for w in warnings)

    def test_list_cell_warns_for_excel(self):
        df = pd.DataFrame({"tags": [["a", "b"], ["c"]], "id": [1, 2]})
        warnings = file_converter.get_conversion_warnings(df, "json", "xlsx")
        assert any("list" in w.lower() or "object" in w.lower() for w in warnings)

    def test_excel_row_limit_warns(self):
        # Build a minimal df and fake its length for the warning check
        import numpy as np
        # Use a real large df would be slow; instead directly test the logic
        df_large = pd.DataFrame({"a": range(10)})
        # Patch len — instead test with actual boundary using a mock-free approach:
        # We check the warning is generated when rows > 1_048_576
        # Since creating 1M rows is slow, we verify the condition logic directly.
        big_df = pd.DataFrame({"a": pd.array(range(1_048_577), dtype="int32")})
        warnings = file_converter.get_conversion_warnings(big_df, "csv", "xlsx")
        assert any("row limit" in w.lower() or "1,048,576" in w for w in warnings)
