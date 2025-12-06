"""
Unit tests for models module.
"""
import pytest
from models import (
    gpt2, bert, gptj, llama, baichuan, t5, 
    chatglm2, internlm, qwen, mistral, support_models
)


class TestModelDefinitions:
    """Test cases for model definitions."""
    
    def test_gpt2_model_defined(self):
        """Test GPT-2 model is defined."""
        assert gpt2 is not None
        assert isinstance(gpt2, str)
        assert "gpt2" in gpt2.lower()
    
    def test_bert_model_defined(self):
        """Test BERT model is defined."""
        assert bert is not None
        assert isinstance(bert, str)
        assert "bert" in bert.lower()
    
    def test_gptj_model_defined(self):
        """Test GPT-J model is defined."""
        assert gptj is not None
        assert isinstance(gptj, str)
        assert "gpt-j" in gptj.lower() or "gptj" in gptj.lower()
    
    def test_llama_model_defined(self):
        """Test LLaMA model is defined."""
        assert llama is not None
        assert isinstance(llama, str)
        assert "llama" in llama.lower()
    
    def test_baichuan_model_defined(self):
        """Test Baichuan model is defined."""
        assert baichuan is not None
        assert isinstance(baichuan, str)
        assert "baichuan" in baichuan.lower()
    
    def test_t5_model_defined(self):
        """Test T5 model is defined."""
        assert t5 is not None
        assert isinstance(t5, str)
        assert "t5" in t5.lower()
    
    def test_chatglm2_model_defined(self):
        """Test ChatGLM2 model is defined."""
        assert chatglm2 is not None
        assert isinstance(chatglm2, str)
        assert "chatglm" in chatglm2.lower()
    
    def test_internlm_model_defined(self):
        """Test InternLM model is defined."""
        assert internlm is not None
        assert isinstance(internlm, str)
        assert "internlm" in internlm.lower()
    
    def test_qwen_model_defined(self):
        """Test Qwen model is defined."""
        assert qwen is not None
        assert isinstance(qwen, str)
        assert "qwen" in qwen.lower()
    
    def test_mistral_model_defined(self):
        """Test Mistral model is defined."""
        assert mistral is not None
        assert isinstance(mistral, str)
        assert "mistral" in mistral.lower()


class TestSupportModels:
    """Test cases for support_models list."""
    
    def test_support_models_exists(self):
        """Test that support_models list exists."""
        assert support_models is not None
        assert isinstance(support_models, list)
    
    def test_support_models_not_empty(self):
        """Test that support_models list is not empty."""
        assert len(support_models) > 0
    
    def test_support_models_contains_all_models(self):
        """Test that support_models contains all defined models."""
        all_models = [gpt2, bert, gptj, llama, baichuan, t5, chatglm2, internlm, qwen, mistral]
        
        for model in all_models:
            assert model in support_models, f"Model {model} not in support_models"
    
    def test_support_models_all_strings(self):
        """Test that all items in support_models are strings."""
        for model in support_models:
            assert isinstance(model, str), f"Model {model} is not a string"
    
    def test_support_models_no_duplicates(self):
        """Test that support_models has no duplicate entries."""
        assert len(support_models) == len(set(support_models)), \
            "support_models contains duplicate entries"
    
    def test_support_models_count(self):
        """Test that support_models has expected number of models."""
        # At least 10 models should be supported
        assert len(support_models) >= 10


class TestModelNamingConventions:
    """Test cases for model naming conventions."""
    
    def test_models_follow_huggingface_format(self):
        """Test that model names follow HuggingFace format."""
        # Most models should have org/model format
        huggingface_models = [
            gpt2, bert, gptj, llama, baichuan, 
            t5, chatglm2, internlm, qwen, mistral
        ]
        
        for model in huggingface_models:
            # Should contain either '/' or be a simple name like 'gpt2'
            assert '/' in model or model.lower() in ['gpt2'], \
                f"Model {model} doesn't follow expected format"
    
    def test_no_empty_model_names(self):
        """Test that no model names are empty."""
        all_models = [gpt2, bert, gptj, llama, baichuan, t5, chatglm2, internlm, qwen, mistral]
        
        for model in all_models:
            assert model, "Found empty model name"
            assert len(model) > 0, "Found empty model name"
    
    def test_model_names_unique(self):
        """Test that all model names are unique."""
        all_models = [gpt2, bert, gptj, llama, baichuan, t5, chatglm2, internlm, qwen, mistral]
        
        assert len(all_models) == len(set(all_models)), \
            "Found duplicate model names"
