"""
Unit tests for dataset_process/kv_template.py module.
"""
import pytest
from dataset_process.kv_template import key2meaning


class TestKVTemplate:
    """Test cases for key-value template definitions."""
    
    def test_key2meaning_exists(self):
        """Test that key2meaning dictionary exists."""
        assert key2meaning is not None
        assert isinstance(key2meaning, dict)
    
    def test_prompt_key(self):
        """Test prompt key exists in template."""
        assert "prompt" in key2meaning
        assert isinstance(key2meaning["prompt"], str)
    
    def test_prompts_key(self):
        """Test prompts key exists in template."""
        assert "prompts" in key2meaning
        assert isinstance(key2meaning["prompts"], str)
    
    def test_ground_truth_key(self):
        """Test ground_truth key exists in template."""
        assert "ground_truth" in key2meaning
        assert isinstance(key2meaning["ground_truth"], str)
    
    def test_triple_keys(self):
        """Test triple-related keys exist in template."""
        assert "triple" in key2meaning
        assert "triple_subject" in key2meaning
        assert "triple_relation" in key2meaning
        assert "triple_object" in key2meaning
    
    def test_all_values_are_strings(self):
        """Test that all values in key2meaning are strings."""
        for key, value in key2meaning.items():
            assert isinstance(value, str), f"Value for key '{key}' is not a string"
    
    def test_required_keys_present(self):
        """Test that all required keys are present."""
        required_keys = [
            "prompt", 
            "prompts", 
            "ground_truth", 
            "triple_subject", 
            "triple_relation", 
            "triple_object"
        ]
        for key in required_keys:
            assert key in key2meaning, f"Required key '{key}' is missing"
