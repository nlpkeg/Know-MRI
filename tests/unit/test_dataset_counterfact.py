"""
Unit tests for dataset_process module.
"""
import pytest
import json
from pathlib import Path

try:
    from dataset_process.counterfact import (
        get_processed_kvs, 
        support_template_keys,
        CounterfactDataset
    )
    COUNTERFACT_AVAILABLE = True
except ImportError:
    COUNTERFACT_AVAILABLE = False


@pytest.mark.skipif(not COUNTERFACT_AVAILABLE, reason="PyTorch/Dataset dependencies not available")
class TestCounterfactDataset:
    """Test cases for CounterfactDataset class."""
    
    def test_get_processed_kvs_basic(self, sample_counterfact_data):
        """Test basic processing of CounterFact sample."""
        result = get_processed_kvs(sample_counterfact_data)
        
        assert "dataset_name" in result
        assert result["dataset_name"] == "CounterFact"
        assert "prompt" in result
        assert result["prompt"] == sample_counterfact_data["prompt"]
    
    def test_get_processed_kvs_prompts(self, sample_counterfact_data):
        """Test prompts key generation."""
        result = get_processed_kvs(sample_counterfact_data, keys=["prompts"])
        
        assert "prompts" in result
        assert isinstance(result["prompts"], list)
        assert len(result["prompts"]) == 1
        assert result["prompts"][0] == sample_counterfact_data["prompt"]
    
    def test_get_processed_kvs_ground_truth(self, sample_counterfact_data):
        """Test ground_truth key mapping."""
        result = get_processed_kvs(sample_counterfact_data, keys=["ground_truth"])
        
        assert "ground_truth" in result
        assert result["ground_truth"] == sample_counterfact_data["target_new"]
    
    def test_get_processed_kvs_triple_subject(self, sample_counterfact_data):
        """Test triple_subject key mapping."""
        result = get_processed_kvs(sample_counterfact_data, keys=["triple_subject"])
        
        assert "triple_subject" in result
        assert result["triple_subject"] == sample_counterfact_data["subject"]
    
    def test_get_processed_kvs_triple_object(self, sample_counterfact_data):
        """Test triple_object key mapping."""
        result = get_processed_kvs(sample_counterfact_data, keys=["triple_object"])
        
        assert "triple_object" in result
        assert result["triple_object"] == sample_counterfact_data["ground_truth"]
    
    def test_get_processed_kvs_all_keys(self, sample_counterfact_data):
        """Test processing with all supported keys."""
        result = get_processed_kvs(sample_counterfact_data, keys=support_template_keys)
        
        assert "prompts" in result
        assert "ground_truth" in result
        assert "triple_subject" in result
        assert "triple_object" in result
        assert "dataset_name" in result
    
    def test_support_template_keys(self):
        """Test that support_template_keys contains expected keys."""
        assert "prompt" in support_template_keys
        assert "prompts" in support_template_keys
        assert "ground_truth" in support_template_keys
        assert "triple_subject" in support_template_keys
        assert "triple_object" in support_template_keys
    
    def test_dataset_loading(self, tmp_path):
        """Test CounterfactDataset loading from file."""
        # Create a temporary dataset file
        dataset_data = [
            {
                "prompt": "Test prompt 1",
                "subject": "Subject 1",
                "target_new": "Target 1",
                "ground_truth": "Truth 1"
            },
            {
                "prompt": "Test prompt 2",
                "subject": "Subject 2",
                "target_new": "Target 2",
                "ground_truth": "Truth 2"
            }
        ]
        
        dataset_file = tmp_path / "test_counterfact.json"
        with open(dataset_file, 'w') as f:
            json.dump(dataset_data, f)
        
        # Load dataset
        dataset = CounterfactDataset(loc=dataset_file)
        
        # Verify
        assert len(dataset) == 2
        assert dataset[0]["prompt"] == "Test prompt 1"
        assert dataset[1]["subject"] == "Subject 2"
    
    def test_dataset_getitem(self, tmp_path):
        """Test dataset __getitem__ method."""
        dataset_data = [{"prompt": "Test", "subject": "S"}]
        
        dataset_file = tmp_path / "test.json"
        with open(dataset_file, 'w') as f:
            json.dump(dataset_data, f)
        
        dataset = CounterfactDataset(loc=dataset_file)
        item = dataset[0]
        
        assert item["prompt"] == "Test"
        assert item["subject"] == "S"
    
    def test_dataset_len(self, tmp_path):
        """Test dataset __len__ method."""
        dataset_data = [{"prompt": f"Test {i}"} for i in range(5)]
        
        dataset_file = tmp_path / "test.json"
        with open(dataset_file, 'w') as f:
            json.dump(dataset_data, f)
        
        dataset = CounterfactDataset(loc=dataset_file)
        assert len(dataset) == 5
