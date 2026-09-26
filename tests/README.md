# ByteMind — Test Suite

## Structure

```
tests/
├── __init__.py
├── test_file_converter.py   — unit tests for bytemind/file_converter.py
├── test_data_health.py      — unit tests for bytemind/data_health.py
├── test_analysis.py         — unit tests for bytemind/analysis.py
├── test_investigation.py    — unit tests for bytemind/investigation.py
└── test_ai_assistant.py     — unit tests for bytemind/ai_assistant.py
```

Run all tests:
```bash
pytest tests/ -v
```
