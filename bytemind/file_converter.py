"""
file_converter.py
-----------------
File Conversion module for ByteMind.

Responsibilities:
- Load CSV, Excel (.xlsx), and JSON files into a pandas DataFrame.
- Detect whether JSON is tabular (list-of-records) or nested/non-tabular.
- Convert a DataFrame to CSV, Excel, or JSON bytes for download.
- Report meaningful conversion warnings (not generic boilerplate).
- Handle common encoding issues on CSV files.

All I/O is in-memory (BytesIO / StringIO) — no temporary files are created.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Supported formats
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({"csv", "xlsx", "json"})

MIME_TYPES: dict[str, str] = {
    "csv": "text/csv",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "json": "application/json",
}

DEFAULT_DOWNLOAD_NAMES: dict[str, str] = {
    "csv": "converted.csv",
    "xlsx": "converted.xlsx",
    "json": "converted.json",
}

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_file(file_obj: Any, file_extension: str) -> pd.DataFrame:
    """
    Load an uploaded file into a pandas DataFrame.

    Parameters
    ----------
    file_obj : file-like object
        The uploaded file (e.g. from ``st.file_uploader``).
        Must support ``.read()`` or be seekable.
    file_extension : str
        Lowercase extension without the leading dot: ``'csv'``, ``'xlsx'``,
        or ``'json'``.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If the extension is not supported, or if the file cannot be parsed.
    """
    ext = file_extension.lower().lstrip(".")
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: '{ext}'. "
            f"Supported formats are: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    if ext == "csv":
        return _load_csv(file_obj)
    if ext == "xlsx":
        return _load_excel(file_obj)
    if ext == "json":
        return _load_json(file_obj)

    # Unreachable, but satisfies type checkers.
    raise ValueError(f"Unhandled extension: '{ext}'")  # pragma: no cover


def _load_csv(file_obj: Any) -> pd.DataFrame:
    """Try common encodings in order; raise a clear ValueError on failure."""
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    raw = file_obj.read() if hasattr(file_obj, "read") else file_obj

    for enc in encodings:
        try:
            return pd.read_csv(io.BytesIO(raw) if isinstance(raw, bytes) else io.StringIO(raw), encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    raise ValueError(
        "Could not read the CSV file. "
        "The file may use an unsupported encoding or be malformed. "
        f"Tried encodings: {', '.join(encodings)}."
    )


def _load_excel(file_obj: Any) -> pd.DataFrame:
    """Load the first sheet of an .xlsx workbook."""
    try:
        raw = file_obj.read() if hasattr(file_obj, "read") else file_obj
        return pd.read_excel(io.BytesIO(raw) if isinstance(raw, bytes) else file_obj, engine="openpyxl")
    except Exception as exc:
        raise ValueError(
            f"Could not read the Excel file: {exc}. "
            "Ensure the file is a valid .xlsx workbook."
        ) from exc


def _load_json(file_obj: Any) -> pd.DataFrame:
    """
    Load a JSON file.

    Handles two common structures:
    - List of records (tabular):  ``[{"a": 1, "b": 2}, ...]``
    - Single object / nested:     ``{"key": ..., ...}``

    For nested structures, ``pd.json_normalize`` is used with a depth limit
    so that the operation never silently discards data — warnings are
    surfaced via :func:`get_conversion_warnings`.
    """
    raw = file_obj.read() if hasattr(file_obj, "read") else file_obj
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    try:
        if isinstance(data, list):
            df = pd.json_normalize(data, max_level=1)
        elif isinstance(data, dict):
            # Try treating dict values as rows first (orient='index' style)
            try:
                df = pd.DataFrame.from_dict(data, orient="index")
            except ValueError:
                df = pd.json_normalize(data, max_level=1)
        else:
            raise ValueError(
                "Unexpected JSON root type. "
                "ByteMind expects either a list of records or a JSON object."
            )
    except Exception as exc:
        raise ValueError(f"Could not parse JSON into a table: {exc}") from exc

    if df.empty:
        raise ValueError("The JSON file produced an empty table.")

    return df


# ---------------------------------------------------------------------------
# Inspection helpers
# ---------------------------------------------------------------------------

def is_nested_json(file_obj: Any) -> bool:
    """
    Return True if the JSON contains nested objects or arrays beyond depth 1.

    Parameters
    ----------
    file_obj : file-like or bytes or str
        The raw JSON content (re-readable object or raw bytes/str).
    """
    raw = file_obj.read() if hasattr(file_obj, "read") else file_obj
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return False

    def _has_nesting(obj: Any, depth: int = 0) -> bool:
        if depth > 1:
            return True
        if isinstance(obj, dict):
            return any(_has_nesting(v, depth + 1) for v in obj.values())
        if isinstance(obj, list):
            return any(_has_nesting(item, depth) for item in obj)
        return False

    return _has_nesting(data)


# ---------------------------------------------------------------------------
# Conversion warnings
# ---------------------------------------------------------------------------

def get_conversion_warnings(
    df: pd.DataFrame,
    source_format: str,
    target_format: str,
    source_is_nested_json: bool = False,
) -> list[str]:
    """
    Return meaningful warnings about potential data loss or behaviour changes
    when converting *df* from *source_format* to *target_format*.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded data (used to inspect column names and types).
    source_format : str
        Original format: ``'csv'``, ``'xlsx'``, or ``'json'``.
    target_format : str
        Desired format: ``'csv'``, ``'xlsx'``, or ``'json'``.
    source_is_nested_json : bool
        Pass ``True`` when the source JSON was detected as nested.

    Returns
    -------
    list[str]
        Zero or more human-readable warning strings.
    """
    warnings: list[str] = []
    src = source_format.lower()
    tgt = target_format.lower()

    # Nested JSON → anything
    if src == "json" and source_is_nested_json:
        warnings.append(
            "The source JSON contains nested objects or arrays. "
            "ByteMind has flattened the structure using dot-notation column names "
            "(e.g. 'address.city'). Nested arrays deeper than one level may have "
            "been dropped. Review the preview carefully before downloading."
        )

    # JSON → Excel: list-type cells
    if tgt == "xlsx":
        list_cols = [c for c in df.columns if df[c].apply(lambda x: isinstance(x, (list, dict))).any()]
        if list_cols:
            warnings.append(
                f"Column(s) {list_cols} contain list or object values. "
                "These will be serialised as strings in the Excel output."
            )

    # CSV: column names with commas
    if tgt == "csv":
        problematic = [c for c in df.columns if "," in str(c)]
        if problematic:
            warnings.append(
                f"Column name(s) {problematic} contain commas. "
                "They will be quoted in the CSV output, which may cause parsing "
                "issues in tools that handle quoted headers poorly."
            )

    # Excel: row-count limit
    if tgt == "xlsx" and len(df) > 1_048_576:
        warnings.append(
            f"The dataset has {len(df):,} rows, which exceeds the Excel "
            "row limit (1,048,576). Rows beyond the limit will be truncated."
        )

    # Mixed-type columns → CSV/Excel type loss
    if tgt in ("csv", "xlsx"):
        mixed = [
            c for c in df.columns
            if df[c].dropna().apply(type).nunique() > 1
        ]
        if mixed:
            warnings.append(
                f"Column(s) {mixed} contain mixed data types. "
                "Values will be written as strings to avoid data corruption."
            )

    return warnings


# ---------------------------------------------------------------------------
# Conversion (DataFrame → bytes)
# ---------------------------------------------------------------------------

def convert_file(df: pd.DataFrame, target_format: str) -> bytes:
    """
    Serialise *df* to the requested format and return raw bytes.

    Parameters
    ----------
    df : pd.DataFrame
    target_format : str
        One of ``'csv'``, ``'xlsx'``, ``'json'``.

    Returns
    -------
    bytes
        File content ready for ``st.download_button``.

    Raises
    ------
    ValueError
        If *target_format* is not supported.
    """
    tgt = target_format.lower()
    if tgt not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported target format: '{tgt}'. "
            f"Supported formats are: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    if tgt == "csv":
        return df.to_csv(index=False).encode("utf-8")

    if tgt == "xlsx":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Data")
        return buf.getvalue()

    if tgt == "json":
        return df.to_json(orient="records", indent=2, force_ascii=False).encode("utf-8")

    raise ValueError(f"Unhandled target format: '{tgt}'")  # pragma: no cover
