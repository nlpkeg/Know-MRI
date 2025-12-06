# Know-MRI Test Suite

This directory contains the comprehensive test suite for the Know-MRI project.

## Overview

The test suite is organized into two main categories:

- **Unit Tests** (`tests/unit/`): Test individual components and modules in isolation
- **Integration Tests** (`tests/integration/`): Test interactions between multiple components

## Test Structure

```
tests/
├── conftest.py                          # Shared fixtures and test configuration
├── unit/                                # Unit tests
│   ├── test_hparams.py                 # Tests for hyperparameter handling
│   ├── test_model_tokenizer.py         # Tests for model type detection
│   ├── test_dataset_counterfact.py     # Tests for CounterFact dataset
│   ├── test_kv_template.py             # Tests for key-value templates
│   ├── test_methods.py                 # Tests for method registration
│   ├── test_models.py                  # Tests for model definitions
│   └── test_result_template.py         # Tests for result template structure
└── integration/                         # Integration tests
    └── test_diagnose.py                # Tests for diagnosis workflow

pytest.ini                               # Pytest configuration
```

## Running Tests

### Prerequisites

Ensure you have all dependencies installed:

```bash
pip install -r requirements.txt
pip install pytest pytest-cov
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

Run only unit tests:
```bash
pytest tests/unit/
```

Run only integration tests:
```bash
pytest tests/integration/
```

### Run with Coverage

```bash
pytest --cov=. --cov-report=html --cov-report=term
```

This will generate a coverage report in `htmlcov/index.html`.

### Run Specific Test Files

```bash
pytest tests/unit/test_hparams.py
pytest tests/unit/test_models.py -v
```

### Run Specific Test Functions

```bash
pytest tests/unit/test_hparams.py::TestHyperParams::test_from_json
pytest -k "test_model_type"
```

## Test Markers

Tests are marked with the following markers:

- `@pytest.mark.unit`: Unit tests (default for tests in `unit/`)
- `@pytest.mark.integration`: Integration tests (default for tests in `integration/`)
- `@pytest.mark.slow`: Tests that take significant time to run

Run tests by marker:
```bash
pytest -m unit
pytest -m integration
pytest -m "not slow"
```

## Writing New Tests

### Unit Test Template

```python
"""
Unit tests for <module_name>.
"""
import pytest
from <module> import <function_or_class>


class Test<ComponentName>:
    """Test cases for <component>."""
    
    def test_<specific_behavior>(self):
        """Test that <specific behavior> works correctly."""
        # Arrange
        input_data = "test"
        
        # Act
        result = function_or_class(input_data)
        
        # Assert
        assert result == expected_output
```

### Using Fixtures

Fixtures are defined in `conftest.py` and are automatically available to all tests:

```python
def test_with_fixture(sample_counterfact_data):
    """Test using a predefined fixture."""
    assert "prompt" in sample_counterfact_data
```

### Common Fixtures

- `sample_counterfact_data`: Sample CounterFact dataset entry
- `sample_processed_kvs`: Processed key-value sample
- `mock_hparams_data`: Mock hyperparameters
- `temp_json_file`: Factory for creating temporary JSON files

## Test Coverage

The test suite covers:

1. **Utility Modules**:
   - HyperParams loading and validation
   - Model type detection
   - Attribute access helpers

2. **Dataset Processing**:
   - Dataset loading and indexing
   - Sample processing and key mapping
   - Template validation

3. **Method Registration**:
   - Method discovery and registration
   - Method interface validation
   - Required attributes checking

4. **Model Definitions**:
   - Model path definitions
   - Support model list validation
   - Naming convention checks

5. **Result Templates**:
   - Result structure validation
   - Image and table format checking

6. **Integration Workflows**:
   - Complete diagnosis pipeline
   - Method lookup and execution
   - Parameter passing

## Continuous Integration

These tests are designed to run in CI/CD pipelines. The pytest configuration in `pytest.ini` ensures:

- Verbose output for debugging
- Short traceback format
- Strict marker enforcement
- Warning suppression for known issues

## Troubleshooting

### Import Errors

If you encounter import errors, ensure you're running tests from the project root:

```bash
cd /path/to/Know-MRI
pytest
```

### Missing Dependencies

Some tests may require optional dependencies. Install them:

```bash
pip install torch transformers
```

### Skipped Tests

Some tests are automatically skipped if required modules are not available:

```python
pytest.skip("Module not available")
```

This is expected behavior and allows the test suite to run partially even when some components are unavailable.

## Best Practices

1. **Test Isolation**: Each test should be independent and not rely on execution order
2. **Clear Naming**: Use descriptive test names that explain what is being tested
3. **AAA Pattern**: Follow Arrange-Act-Assert pattern in tests
4. **Mock External Dependencies**: Use mocks for models, APIs, and file I/O
5. **Test Edge Cases**: Include tests for error conditions and boundary values

## Contributing

When adding new features:

1. Write tests first (TDD approach recommended)
2. Ensure all tests pass before submitting
3. Maintain test coverage above 80%
4. Add docstrings to test functions
5. Update this README if adding new test categories

## Contact

For questions about the test suite, please open an issue on the GitHub repository.
