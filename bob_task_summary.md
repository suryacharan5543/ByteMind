# ByteMind — Bob Task Summary

This file documents Bob's (IBM Bob 2.0) contribution to each major development task.
It is updated at the end of every task session.

---

## Task 1 — Project Initialisation

**Date:** 2026-09-26
**Status:** ✅ Complete

### What Was Done

Bob inspected the empty workspace and bootstrapped the entire ByteMind project from scratch.

#### Project Structure Created

```
ByteMind/
├── app.py                      # Streamlit entry point with full Home UI
├── requirements.txt            # Minimal, intentional dependency list
├── README.md                   # High-level project documentation
├── bob_task_summary.md         # This file
│
├── bytemind/
│   ├── __init__.py             # Package marker + version string
│   ├── file_converter.py       # Placeholder: load_file, convert_file, get_conversion_warnings
│   ├── data_health.py          # Placeholder: run_health_check, get_quality_summary
│   ├── analysis.py             # Placeholder: descriptive_stats, correlation, category, time-series, charts
│   ├── investigation.py        # Placeholder: generate_findings, rank_findings
│   └── ai_assistant.py         # Placeholder: explain_findings, get_suggested_questions
│
└── tests/
    ├── __init__.py
    ├── README.md
    ├── test_file_converter.py   # Verifies placeholder contract (NotImplementedError)
    ├── test_data_health.py
    ├── test_analysis.py
    ├── test_investigation.py
    └── test_ai_assistant.py
```

#### Key Decisions

| Decision | Rationale |
|---|---|
| All feature functions raise `NotImplementedError` | Establishes a clear, testable placeholder contract before implementation begins |
| Tests assert `NotImplementedError` | Confirms the module is importable and the public API surface is correct |
| No fake AI stubs | Prevents confusion between real and simulated functionality |
| API key via env var `BYTEMIND_LLM_API_KEY` | Security first — no secrets ever hard-coded |
| Minimal `requirements.txt` | Avoids unnecessary bloat; dependencies added only when features are built |
| `openpyxl` included now | Required by pandas for Excel support; zero overhead |

#### Streamlit UI (Home Page)

- ByteMind name and tagline header
- 7-step workflow visualised as a card row
- Two audience cards (Analyst / Business Owner)
- Feature roadmap status table
- Sidebar navigation for all 6 planned pages (5 show "Coming soon")

#### Tests Run

All 10 placeholder tests pass:

```
tests/test_file_converter.py    3 passed
tests/test_data_health.py       2 passed
tests/test_analysis.py          5 passed
tests/test_investigation.py     2 passed
tests/test_ai_assistant.py      3 passed
```

---

## Task 2 — File Conversion

**Date:** 2026-09-26
**Status:** ✅ Complete

### What Was Done

Bob implemented the File Conversion feature end-to-end, replacing the Task 1 placeholder in `bytemind/file_converter.py` and wiring it into the Streamlit UI.

#### Files Changed

| File | Change |
|---|---|
| `bytemind/file_converter.py` | Full implementation (replaced placeholder) |
| `app.py` | Upload & Convert page fully wired; Home status table updated |
| `tests/test_file_converter.py` | 35 real unit tests (replaced placeholder tests) |
| `README.md` | Feature status and project structure updated |
| `bob_task_summary.md` | This entry |

#### What Was Implemented

**`bytemind/file_converter.py`**

| Function | Description |
|---|---|
| `load_file(file_obj, ext)` | Dispatches to CSV / Excel / JSON loaders |
| `_load_csv(file_obj)` | Tries utf-8, utf-8-sig, latin-1, cp1252 in order; raises `ValueError` on parse failure |
| `_load_excel(file_obj)` | Loads first sheet via openpyxl; raises `ValueError` on failure |
| `_load_json(file_obj)` | Handles list-of-records and dict roots; uses `pd.json_normalize` for nested JSON |
| `is_nested_json(file_obj)` | Recursive depth check — returns `True` if nesting exceeds depth 1 |
| `get_conversion_warnings(df, src, tgt, nested)` | Returns meaningful, specific warnings (nested flattening, list cells, comma column names, Excel row limit, mixed types) |
| `convert_file(df, target_format)` | Serialises DataFrame to CSV, Excel (.xlsx), or JSON bytes in-memory |
| `MIME_TYPES`, `DEFAULT_DOWNLOAD_NAMES` | Constants used by the Streamlit download button |

**`app.py` — Upload & Convert page**

1. `st.file_uploader` accepting `.csv`, `.xlsx`, `.json`
2. Detected format displayed to user
3. DataFrame loaded and shown: row/column metrics, column-types expander, 50-row preview
4. Nested-JSON detection before format selection
5. Target format selector (excludes same-format conversion)
6. Conversion warnings displayed as `st.warning` blocks when present
7. `st.download_button` with correct MIME type and filename

#### Important Technical Decisions

| Decision | Rationale |
|---|---|
| All I/O via `BytesIO` / `StringIO` — no temp files | Keeps the server stateless; nothing persists between sessions |
| Encoding fallback chain (utf-8 → utf-8-sig → latin-1 → cp1252) | Covers the vast majority of real-world CSV encodings without crashing |
| `pd.json_normalize(max_level=1)` | Flattens one level safely; deeper nesting is flagged via warning rather than silently dropped |
| `is_nested_json` uses recursive depth check, not just `json_normalize` | Allows accurate warning generation *before* conversion happens |
| `get_conversion_warnings` uses `source_is_nested_json` flag | Avoids re-parsing the file; caller passes the pre-computed result |
| Same-format conversion excluded from UI selector | Prevents unnecessary no-op conversions |

#### Tests Performed

All tests use in-memory data — no disk writes.

```
tests/test_file_converter.py    35 passed
tests/test_data_health.py        2 passed  (placeholder contract unchanged)
tests/test_analysis.py           5 passed  (placeholder contract unchanged)
tests/test_investigation.py      2 passed  (placeholder contract unchanged)
tests/test_ai_assistant.py       3 passed  (placeholder contract unchanged)
─────────────────────────────────────────
Total                           47 passed
```

Tests cover: CSV loading, Excel loading, JSON loading (flat and nested), unsupported format errors, malformed CSV (unterminated quote), invalid JSON, empty JSON, `is_nested_json` (flat, nested, deeply nested, invalid), CSV/Excel/JSON conversion round-trips, no-index CSV output, JSON orient=records, unsupported conversion target, end-to-end CSV→JSON and CSV→Excel, all warning conditions.

#### Known Limitations

- Only the first sheet of a multi-sheet Excel workbook is loaded. Multi-sheet support is deferred.
- JSON arrays nested deeper than one level are flattened with dot-notation; data in arrays-within-arrays is not preserved. A warning is shown.
- Very large files (>200 MB) are loaded entirely into memory. Streaming is not implemented.

---

## Task 3 — Data Health Check

**Date:** 2026-09-26
**Status:** ✅ Complete

### What Was Done

Bob implemented the full Data Health Check feature and wired it into the Streamlit app with shared session-state dataset management.

#### Files Changed

| File | Change |
|---|---|
| `bytemind/data_health.py` | Full implementation (replaced placeholder) |
| `app.py` | Session-state shared dataset; Data Health page fully wired; Home status updated |
| `tests/test_data_health.py` | 57 real unit tests (replaced placeholder tests) |
| `README.md` | Feature status and test counts updated |
| `bob_task_summary.md` | This entry |

#### Data Health Checks Implemented

| Function | Check |
|---|---|
| `get_overview` | Rows, columns, numeric/categorical/datetime column breakdown, memory usage |
| `get_missing_values` | Per-column missing count, percentage, substantial flag (≥5%) |
| `get_duplicates` | Exact duplicate-row count and percentage |
| `get_dtype_info` | Per-column dtype, kind (numeric/datetime/categorical), date-column heuristic |
| `get_unique_counts` | Per-column unique count, percentage, high-cardinality flag (≥90% unique) |
| `get_consistency_issues` | Empty strings, whitespace-only values, constant columns, mixed types, infinite values |
| `get_outliers` | IQR-based potential outliers per numeric column (Q1, Q3, IQR, fences, count, %) |
| `compute_health_score` | Transparent 0–100 indicator with 5 documented factors and weights |
| `get_quality_summary` | Structured good/warnings/attention bullet list |
| `run_health_check` | Orchestrates all checks into a single report dict |

#### Health Score — Documented Factors

| Factor | Weight | Measures |
|---|---|---|
| Missing values | 30 pts | Fraction of columns with ≥5% missing |
| Duplicates | 15 pts | Duplicate-row percentage |
| Consistency | 25 pts | Fraction of columns with any issue |
| Outliers | 15 pts | Fraction of numeric cols with ≥5% potential outliers |
| Cardinality | 15 pts | Fraction of non-numeric columns with ≥90% unique values |

#### Session State Integration

- `st.session_state["dataset"]` and `st.session_state["dataset_name"]` are set on any page that loads a file.
- The Upload & Convert page now stores the loaded DataFrame in session state.
- The Data Health page uses `_shared_upload_widget()` which both accepts a new upload and falls through to the already-loaded dataset — the user never needs to re-upload.
- The sidebar shows the active dataset name and shape at all times.

#### Important Technical Decisions

| Decision | Rationale |
|---|---|
| Data Health operates independently | User can go directly to 🏥 Data Health without using File Conversion |
| Session state for shared dataset | Avoids re-upload; consistent with future Analysis/Investigation pages |
| `is_string_col` check covers `object` AND `StringDtype` | Pandas 2.x uses Arrow-backed StringDtype; checking only `dtype == object` missed string columns |
| Empty-row guard in `get_consistency_issues` | Arrow StringDtype `.apply().sum()` on empty series returns `""` not 0 — skip loop body early |
| `pd.to_datetime(str(val))` without `infer_datetime_format` | Argument deprecated and removed in pandas 2.x |
| Outliers labelled "potential" throughout | IQR flags are a starting point; domain knowledge is needed to judge whether a value is wrong |
| Health score documented inline | Weights and thresholds are constants at the top of the module, not magic numbers |

#### Tests Performed

```
tests/test_data_health.py     57 passed
tests/test_file_converter.py  35 passed  (Task 2 — unchanged)
tests/test_analysis.py         5 passed  (placeholder)
tests/test_investigation.py    2 passed  (placeholder)
tests/test_ai_assistant.py     3 passed  (placeholder)
──────────────────────────────────────────
Total                         102 passed
```

Test classes cover: overview, missing values, duplicates, dtype info, unique counts, consistency issues (empty strings, whitespace, constant columns, mixed types, inf), outliers (detected, skipped, zero-IQR, few-values), health score (perfect score, range, dirty<clean, components, grade mapping), quality summary, and full integration (all keys, None input, empty DF, single row).

#### Known Limitations

- Date-column detection is heuristic (≥80% of sample values parse as dates). It may produce false positives for numeric columns that happen to look like years.
- The health score is a ByteMind transparency indicator only — it is not a universally calibrated metric.
- Consistency checks operate without business-domain knowledge; they detect structural issues, not semantic ones.
- The `_clean_df` test fixture scores ~92.5/100 (not 100) because its text columns have 100% unique cardinality, which the cardinality factor correctly penalises. This is expected and documented.

---

## Task 4 — Automatic Data Analysis (EDA)

**Date:** 2026-09-26
**Status:** ✅ Complete

### What Was Done

Bob implemented the full Automatic Data Analysis module, wired it into a six-tab Streamlit page, and replaced the Task 1 placeholder tests with 77 real unit tests.

#### Files Changed

| File | Change |
|---|---|
| `bytemind/analysis.py` | Full implementation (replaced placeholder) |
| `app.py` | Analysis page (6 tabs) + import added + Home roadmap updated |
| `tests/test_analysis.py` | 77 real unit tests (replaced placeholder tests) |
| `README.md` | Feature status, project structure, test counts updated |
| `bob_task_summary.md` | This entry |

#### Analysis Functions Implemented

| Function | What it does |
|---|---|
| `get_overview` | Rows, cols, numeric/categorical/datetime breakdown, missing summary |
| `run_descriptive_stats` | Per numeric column: count, mean, median, std, min, Q1, Q3, max (inf/NaN excluded) |
| `run_correlation_analysis` | Pearson matrix, notable pairs above configurable threshold, sorted by \|r\| |
| `get_usable_categorical_cols` | Excludes numeric, datetime, constant, and high-cardinality (>50% unique) columns |
| `run_category_analysis` | Top-N frequency table with count and % for a categorical column |
| `run_grouped_summary` | mean/sum/count of a numeric column grouped by a categorical column |
| `detect_datetime_columns` | Returns already-parsed datetime columns |
| `try_parse_dates` | Attempts to parse a string column as datetime (≥80% valid required) |
| `run_time_series_analysis` | Aggregates a numeric measure by daily or monthly period; auto-selects frequency |
| `run_top_bottom` | Top and bottom N rows by a numeric column, with optional label column |
| `generate_findings` | Factual, data-derived findings: missing quality, numeric ranges, correlation, top category, trend peak |
| `build_histogram` | Plotly histogram for a numeric column |
| `build_box_plot` | Plotly box plot for a numeric column |
| `build_correlation_heatmap` | Plotly heatmap from a correlation matrix |
| `build_bar_chart` | Horizontal Plotly bar chart |
| `build_line_chart` | Plotly line chart with markers |

#### Streamlit Analysis Page (6 Tabs)

| Tab | Content |
|---|---|
| 📐 Descriptive Stats | Stats table + column explorer (histogram + box plot) |
| 🔗 Correlations | Pearson heatmap + notable-pairs table with anti-causation notes |
| 🏷️ Categories | Frequency table + bar chart + grouped summary (mean/sum/count) |
| 📅 Time Trends | Date column selector, numeric measure, daily/monthly line chart |
| 🏆 Top / Bottom | Configurable N, optional label column, side-by-side tables |
| 💡 Auto Findings | Expandable factual findings with kind icons |

#### Important Technical Decisions

| Decision | Rationale |
|---|---|
| `generate_findings` never uses "caused" / "because of" | Anti-causation constraint enforced in tests and code |
| Correlation notes include "not a cause-effect relationship" | Explicit disclaimer in every notable-pair note |
| `get_usable_categorical_cols` filters high-cardinality columns | Prevents meaningless analysis on ID columns |
| `run_time_series_analysis` auto-selects daily vs monthly | ≥90 distinct dates → monthly avoids noisy daily charts |
| `"ME"` as monthly frequency alias | `"M"` deprecated in pandas 2.2; `"ME"` (month-end) is the current alias |
| Division-by-zero guard in `get_overview` | Empty DataFrames have 0 rows × 0 cols = 0 total_cells |
| Plotly imported inside `try/except` at module level | Module remains importable and testable without a display environment |

#### Tests Performed

```
tests/test_analysis.py        77 passed   ← Task 4 (new)
tests/test_data_health.py     57 passed   ← Task 3 (unchanged)
tests/test_file_converter.py  35 passed   ← Task 2 (unchanged)
tests/test_investigation.py    2 passed   ← placeholder
tests/test_ai_assistant.py     3 passed   ← placeholder
──────────────────────────────────────────
Total                         179 passed
```

Test classes cover: overview (incl. empty df), descriptive stats (inf/NaN exclusion, all-NaN col, no numeric cols), correlation (perfect positive/negative, threshold, constant-col exclusion, causation disclaimer, sorted output), usable categorical cols (ID exclusion, constant exclusion), category analysis (counts, pct, top-N), grouped summary (mean/sum/count, invalid agg, sorted), datetime detection and date parsing (success, failure, missing col), time series (daily/monthly freq, string date, unparseable, too-few-rows), top/bottom (correct order, label col, inf exclusion, empty-after-drop), findings (keys, no causation, empty df, category/quality kinds), chart builders (smoke tests for all 5 chart types), edge cases.

#### Known Limitations

- Date-trend analysis uses sum aggregation only (not mean/count). Future tasks can add aggregation selection.
- Correlation uses Pearson only (linear). Spearman/Kendall could be added later.
- `generate_findings` reports at most the first 3 numeric columns and first 2 categorical columns to keep the output concise.
- Time-series frequency auto-selection threshold (90 distinct dates) is a constant, not user-configurable in this release.

---

## Task 4 — Bug Fix: Dataset Persistence Across Pages

**Date:** 2026-09-26
**Status:** ✅ Complete

### Root Cause

`_shared_upload_widget` in [`app.py`](app.py) always rendered `st.file_uploader`. Streamlit resets widget state when the user navigates between pages, so the uploader appeared empty on return. The DataFrame was still correctly stored in `st.session_state["dataset"]` — the function would reach the `if st.session_state["dataset"] is not None: return True` guard and continue correctly — but the empty file-uploader widget remained visible with no indication that a dataset was already loaded, making it look like the data was gone.

### Fix

Rewrote `_shared_upload_widget` with two distinct branches:

1. **Dataset already in session state** — skip the primary uploader entirely; show a green `st.success` banner ("✅ Dataset loaded: filename — N rows × M cols") and a collapsed "🔄 Replace dataset" expander containing a secondary uploader. Returns `True` immediately.
2. **No dataset in session state** — show the primary uploader as before. Returns `True` after a successful load, `False` if nothing is uploaded.

This is the minimal change that solves the UX problem without touching any analysis, health, or conversion logic.

### Files Changed

| File | Change |
|---|---|
| `app.py` | `_shared_upload_widget` rewritten — dataset-present branch added |
| `bob_task_summary.md` | This entry |

### Test Results

No existing tests were broken. The fix is in UI rendering logic only; the underlying `file_converter.load_file` and session-state key behaviour are unchanged.

```
179 passed, 4 warnings
```

---

## Task 5 — Investigation Engine

**Date:** 2026-09-26
**Status:** ✅ Complete

### What Was Done

Bob implemented the full Investigation Engine, replacing the Task 1 placeholder in `bytemind/investigation.py`, wiring it into a new live Investigation page in `app.py`, and writing 45 unit tests.

#### Files Changed

| File | Change |
|---|---|
| `bytemind/investigation.py` | Full implementation (replaced placeholder) |
| `app.py` | Investigation page live; import added; Home roadmap updated |
| `tests/test_investigation.py` | 45 real unit tests (replaced placeholder tests) |
| `README.md` | Feature status and test counts updated |
| `bob_task_summary.md` | This entry |

#### Investigation Capabilities Implemented

| Detector | What it detects |
|---|---|
| `_correlation_findings` | Strong Pearson associations (|r| ≥ 0.5) between numeric columns |
| `_distribution_findings` | Highly variable numeric columns (CV ≥ 50%) with optional outlier note |
| `_group_variation_findings` | Numeric columns whose mean varies ≥ 2× across categorical groups |
| `_top_category_findings` | Single value dominating ≥ 50% of a column; top-3 covering ≥ 80% |
| `_trend_findings` | Upward/downward/stable trends in numeric columns over datetime columns |

Each finding provides:
- **title** — short, data-specific headline
- **finding** — factual observation with numbers
- **why** — why it may deserve investigation (hedged language: "associated with", "may indicate")
- **next_step** — one concrete suggested action
- **questions** — 2–4 data-specific follow-up questions
- **columns** — which columns are involved
- **kind** — categorisation for icon display in UI

#### Design Constraints Enforced

| Constraint | How enforced |
|---|---|
| No causation claims | `test_correlation_finding_no_causation` + `TestAntiCausation` class — `caused`, `because of`, `leads to`, `results in` are forbidden in all finding fields |
| No assumed column names | Works with arbitrary datasets — no hard-coded "revenue", "profit", "discount" etc. |
| No numerical importance scores | Findings are not sorted by an opaque score; order reflects detection sequence |
| Graceful edge cases | `generate_findings` returns `[]` on empty/null/no-numeric DataFrames; never crashes |
| Cap at MAX_FINDINGS=10 | UI stays readable regardless of dataset complexity |

#### Tests Performed

```
tests/test_investigation.py    45 passed   ← Task 5 (new)
tests/test_analysis.py         77 passed   ← Task 4 (unchanged)
tests/test_data_health.py      57 passed   ← Task 3 (unchanged)
tests/test_file_converter.py   35 passed   ← Task 2 (unchanged)
tests/test_ai_assistant.py      3 passed   ← placeholder
──────────────────────────────────────────
Total                          224 passed
```

Test classes cover: return shape and required keys, empty/null/non-DataFrame inputs, MAX_FINDINGS cap, unique titles, correlation detection (positive/negative, no-causation, column references in questions), distribution variability (CV threshold, constant column not flagged), group variation (10× ratio detected, similar groups not flagged, no categorical = no finding), category dominance (≥50% dominant, top-3 concentration), trend direction over datetime columns, follow-up question structure (non-empty strings, ≤4 per finding, data-specific), anti-causation enforcement across all datasets, edge cases (no numeric, single numeric, NaN/inf, single row, tiny dataset).

#### Known Limitations

- Group variation finder checks the first 3 categorical columns × first 4 numeric columns to keep runtime bounded.
- Trend finder only uses `sum` aggregation (inherited from `analysis.run_time_series_analysis`).
- `generate_findings` is called fresh each page load — results are not cached. For very large datasets (>100k rows), this may be slow. Caching is deferred.
- `rank_findings` placeholder function is not re-used — the original placeholder is removed from the public API and findings are returned in detection order.

---

## Task 6 — AI Assistant (Google Gemini)

**Date:** 2026-09-26
**Status:** ✅ Complete

### What Was Done

Bob implemented the full AI Assistant module using Google Gemini, wired it into the `🤖 AI Assistant` Streamlit page, added `google-generativeai` to `requirements.txt`, and replaced the 3 placeholder tests with 52 real unit tests.

#### Files Changed

| File | Change |
|---|---|
| `bytemind/ai_assistant.py` | Full implementation (replaced placeholder) |
| `app.py` | AI Assistant page live; import added; Home roadmap updated to ✅ |
| `requirements.txt` | `google-generativeai>=0.7.0` added |
| `tests/test_ai_assistant.py` | 52 real unit tests (replaced 3 placeholder tests) |
| `README.md` | AI section, feature status, test counts, stack table updated |
| `bob_task_summary.md` | This entry |

#### What Was Implemented

**`bytemind/ai_assistant.py`**

| Symbol | Description |
|---|---|
| `ConfigurationError` | Raised when `GEMINI_API_KEY` env var is absent or empty |
| `AIAssistantError` | Raised on any Gemini API-level failure |
| `build_context(dataset_name, overview, findings, analysis_results)` | Assembles a compact plain-text summary — never includes the raw DataFrame |
| `explain_findings(findings, analysis_results, mode, dataset_name, overview)` | Sends the context to Gemini and returns the explanation text |
| `get_suggested_questions(findings, mode, dataset_name, overview)` | Asks Gemini for follow-up questions; parses bullet-point lines into a list |
| `_get_api_key()` | Reads `GEMINI_API_KEY` from env; raises `ConfigurationError` if absent/empty |
| `_call_gemini(api_key, prompt)` | Calls `google.generativeai`; wraps all errors as `AIAssistantError` |

**Two explanation modes**

| Mode | Behaviour |
|---|---|
| `analyst` | Technical language, column references, statistical context, suggested analyses; explicitly distinguishes association from causation |
| `business` | Plain business language, no statistical jargon, practical questions; does not claim correlation proves causation |

**Streamlit page wiring (`app.py` — `🤖 AI Assistant`)**

1. `GEMINI_API_KEY` check on page entry — shows a clear setup message and stops (no crash) if key is absent
2. Shared upload widget — dataset loaded from session state, same as all other pages
3. Mode radio button: `📈 Junior Data Analyst` / `🏪 Business Owner`
4. "Explain findings" primary button — calls `explain_findings()`, renders Gemini response as markdown
5. "Generate questions" secondary button — calls `get_suggested_questions()`, renders as bullet list
6. All errors (ConfigurationError, AIAssistantError, unexpected) are caught and shown with `st.error()`; app remains usable
7. "What is sent to Gemini?" expander — renders `build_context()` output as plain text for full transparency

#### Design Constraints Enforced

| Constraint | How enforced |
|---|---|
| Raw dataset never sent to LLM | `build_context()` only accepts `overview` dict and `findings` list; `test_raw_dataframe_not_in_prompt` asserts no DataFrame repr in prompt |
| No hard-coded API key | `_get_api_key()` reads env var only; `test_raises_configuration_error_when_key_absent` covers missing-key path |
| No real API calls in tests | All tests mock `_call_gemini` or `_get_api_key` via `unittest.mock.patch`; verified by checking no network activity |
| App usable without key | API key check uses `st.stop()` with an informative warning — other pages are unaffected |
| No causation claims in prompts | System prompts explicitly instruct Gemini: "Never claim a correlation proves causation" |

#### Tests Performed

```
tests/test_ai_assistant.py     52 passed   ← Task 6 (new — replaced 3 placeholders)
tests/test_investigation.py    45 passed   ← Task 5 (unchanged)
tests/test_analysis.py         77 passed   ← Task 4 (unchanged)
tests/test_data_health.py      57 passed   ← Task 3 (unchanged)
tests/test_file_converter.py   35 passed   ← Task 2 (unchanged)
──────────────────────────────────────────
Total                          266 passed, 4 warnings (external library warnings only)
```

The 4 warnings are unchanged from Task 5:
- 2× pandas `UserWarning` from `test_try_parse_dates_failure` intentionally parsing unparseable data
- 2× numpy `RuntimeWarning` from quantile interpolation on empty arrays in dirty-data fixture

#### Known Limitations

- `google-generativeai` must be installed (`pip install -r requirements.txt`). The import is deferred to `_call_gemini()` so all other features remain importable even if the package is absent — a clear `AIAssistantError` is raised only if the AI page is actually used.
- Gemini response quality depends on the quality of investigation findings fed to it. On datasets with no strong patterns, the AI has little to explain.
- `get_suggested_questions()` relies on Gemini returning bullet-point formatted text. If the model deviates, the result may be a single-item list. The page handles this gracefully.
- Rate limits and quota errors from the Gemini API are caught and shown as `st.error()` messages; the user is directed to check their key and quota.
