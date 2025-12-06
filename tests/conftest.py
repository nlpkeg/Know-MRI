"""
Pytest configuration and shared fixtures for Know-MRI test suite.
"""
import pytest
import sys
import os
from pathlib import Path

# Add the root directory to the path for imports
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))


@pytest.fixture
def sample_counterfact_data():
    """Fixture providing sample CounterFact dataset entry."""
    return {
        "case_id": 0,
        "requested_rewrite": {
            "prompt": "The headquarters of {} is located in",
            "subject": "Apple Inc.",
            "target_new": "Seattle",
            "target_true": "Cupertino"
        },
        "prompt": "The headquarters of Apple Inc. is located in",
        "subject": "Apple Inc.",
        "target_new": "Seattle",
        "target_true": "Cupertino",
        "ground_truth": "Cupertino"
    }


@pytest.fixture
def sample_processed_kvs():
    """Fixture providing processed key-value sample."""
    return {
        "prompt": "The headquarters of Apple Inc. is located in",
        "prompts": ["The headquarters of Apple Inc. is located in"],
        "ground_truth": "Seattle",
        "triple_subject": "Apple Inc.",
        "triple_object": "Cupertino",
        "dataset_name": "CounterFact",
        "dataset_type": ""
    }


@pytest.fixture
def mock_hparams_data():
    """Fixture providing mock hyperparameters."""
    return {
        "lr_scale": 0.5,
        "n_toks": 10,
        "model_path": "test_model",
        "refine": True,
        "batch_size": 32,
        "steps": 100,
        "adaptive_threshold": 0.3,
        "p": 0.5
    }


@pytest.fixture
def temp_json_file(tmp_path):
    """Fixture providing a temporary JSON file path."""
    def _create_json(data):
        import json
        file_path = tmp_path / "test_data.json"
        with open(file_path, 'w') as f:
            json.dump(data, f)
        return file_path
    return _create_json
