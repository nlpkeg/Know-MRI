"""
Unit tests for util/hparams.py module.
"""
import pytest
import json
from pathlib import Path
from dataclasses import dataclass
from util.hparams import HyperParams


class TestHyperParams:
    """Test cases for HyperParams class."""
    
    def test_hyperparams_instantiation(self):
        """Test basic HyperParams instantiation."""
        @dataclass
        class TestParams(HyperParams):
            lr: float = 0.001
            epochs: int = 10
        
        params = TestParams(lr=0.01, epochs=20)
        assert params.lr == 0.01
        assert params.epochs == 20
    
    def test_from_json(self, tmp_path, mock_hparams_data):
        """Test loading HyperParams from JSON file."""
        @dataclass
        class TestParams(HyperParams):
            lr_scale: float = None
            n_toks: int = None
            model_path: str = None
            refine: bool = None
            batch_size: int = None
            steps: int = None
            adaptive_threshold: float = None
            p: float = None
        
        # Create temp JSON file
        json_file = tmp_path / "test_params.json"
        with open(json_file, 'w') as f:
            json.dump(mock_hparams_data, f)
        
        # Load from JSON
        params = TestParams.from_json(json_file)
        
        # Verify values
        assert params.lr_scale == 0.5
        assert params.n_toks == 10
        assert params.model_path == "test_model"
        assert params.refine is True
        assert params.batch_size == 32
        assert params.steps == 100
        assert params.adaptive_threshold == 0.3
        assert params.p == 0.5
    
    def test_from_json_missing_file(self):
        """Test error handling when JSON file doesn't exist."""
        @dataclass
        class TestParams(HyperParams):
            lr: float = None
        
        with pytest.raises(FileNotFoundError):
            TestParams.from_json("nonexistent_file.json")
    
    def test_from_json_invalid_json(self, tmp_path):
        """Test error handling for invalid JSON."""
        @dataclass
        class TestParams(HyperParams):
            lr: float = None
        
        # Create invalid JSON file
        json_file = tmp_path / "invalid.json"
        with open(json_file, 'w') as f:
            f.write("{ invalid json }")
        
        with pytest.raises(json.JSONDecodeError):
            TestParams.from_json(json_file)
    
    def test_default_values(self):
        """Test that default values are properly set."""
        @dataclass
        class TestParams(HyperParams):
            lr: float = 0.001
            epochs: int = 10
            name: str = "default"
        
        params = TestParams()
        assert params.lr == 0.001
        assert params.epochs == 10
        assert params.name == "default"
