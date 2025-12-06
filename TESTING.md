# Testing Quick Start Guide

This guide will help you quickly get started with running tests in Know-MRI.

## Installation

1. **Install pytest** (if not already installed):
```bash
pip install pytest pytest-cov pytest-mock
```

2. **Install project dependencies** (optional, for full test coverage):
```bash
pip install -r requirements.txt
```

## Running Tests

### Quick Test Run

Run all tests:
```bash
pytest
```

Run tests with more detail:
```bash
pytest -v
```

Run tests quietly:
```bash
pytest -q
```

### Run Specific Test Categories

Unit tests only:
```bash
pytest tests/unit/
```

Integration tests only:
```bash
pytest tests/integration/
```

Specific test file:
```bash
pytest tests/unit/test_models.py
```

### Run Tests with Coverage

```bash
pytest --cov=. --cov-report=html
```

Then open `htmlcov/index.html` in your browser.

## Test Results Explained

- **PASSED (.)**: Test ran successfully
- **SKIPPED (s)**: Test skipped (usually due to missing dependencies)
- **FAILED (F)**: Test failed (indicates a problem)

### Example Output

```
tests/unit/test_models.py::TestModelDefinitions::test_gpt2_model_defined PASSED [ 67%]
tests/unit/test_models.py::TestModelDefinitions::test_bert_model_defined PASSED [ 68%]
...
============================================ 53 passed, 36 skipped in 0.14s ============================================
```

This shows:
- 53 tests passed successfully
- 36 tests were skipped (because PyTorch/transformers aren't installed)
- All tests completed in 0.14 seconds

## Understanding Skipped Tests

Many tests are skipped when heavy dependencies (PyTorch, transformers) are not installed. This is intentional and allows:

1. **Fast CI/CD**: Run lightweight tests without installing large ML libraries
2. **Flexibility**: Developers can run relevant tests even without full environment
3. **Gradual Setup**: Install dependencies as needed

To run all tests including skipped ones, install the full requirements:
```bash
pip install torch transformers
pytest
```

## Common Test Commands

| Command | Purpose |
|---------|---------|
| `pytest` | Run all tests |
| `pytest -v` | Verbose output |
| `pytest -k "test_models"` | Run tests matching pattern |
| `pytest --tb=short` | Show shorter tracebacks |
| `pytest --lf` | Run last failed tests |
| `pytest --failed-first` | Run failed tests first, then others |

## Writing New Tests

See `tests/README.md` for detailed instructions on writing tests.

Quick template:
```python
import pytest

def test_my_feature():
    """Test that my feature works."""
    result = my_function()
    assert result == expected_value
```

## Troubleshooting

### ImportError: No module named 'X'

Install the missing module:
```bash
pip install X
```

Or run tests that don't require it:
```bash
pytest tests/unit/test_models.py  # Doesn't require PyTorch
```

### Tests Taking Too Long

Run a subset:
```bash
pytest tests/unit/  # Skip integration tests
pytest -k "not slow"  # Skip slow tests
```

### All Tests Failing

1. Check you're in the project root directory
2. Ensure pytest is installed: `pip install pytest`
3. Check Python version: `python --version` (requires Python 3.8+)

## CI/CD Integration

The test suite is designed to work in CI/CD environments. Example GitHub Actions workflow:

```yaml
- name: Run tests
  run: |
    pip install pytest pytest-cov
    pytest --cov=. --cov-report=xml
```

## Getting Help

- Full documentation: See `tests/README.md`
- Report issues: Open an issue on GitHub
- Ask questions: Check existing issues or open a new one
