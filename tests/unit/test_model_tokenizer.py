"""
Unit tests for util/model_tokenizer.py module.
"""
import pytest

try:
    from util.model_tokenizer import model_type, get_attributes
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch and transformers not available")
class TestModelType:
    """Test cases for model type detection."""
    
    def test_bert_detection(self):
        """Test BERT model detection."""
        assert model_type("bert-base-uncased") == "bert"
        assert model_type("bert-large-cased") == "bert"
        assert model_type("google-bert/bert-base-uncased") == "bert"
    
    def test_gpt2_detection(self):
        """Test GPT-2 model detection."""
        assert model_type("gpt2") == "gpt2"
        assert model_type("gpt2-medium") == "gpt2"
        assert model_type("openai-community/gpt2-xl") == "gpt2"
    
    def test_gptj_detection(self):
        """Test GPT-J model detection."""
        assert model_type("gpt-j-6b") == "gptj"
        assert model_type("EleutherAI/gpt-j-6b") == "gptj"
        assert model_type("gptj-custom") == "gptj"
    
    def test_gpt_neo_detection(self):
        """Test GPT-Neo model detection."""
        assert model_type("EleutherAI/gpt-neo-125M") == "gpt_neo"
        assert model_type("EleutherAI/gpt-neo-1.3B") == "gpt_neo"
        assert model_type("EleutherAI/gpt-neo-2.7B") == "gpt_neo"
    
    def test_t5_detection(self):
        """Test T5 model detection."""
        assert model_type("t5-small") == "t5"
        assert model_type("google/flan-t5-large") == "t5"
    
    def test_llama_detection(self):
        """Test LLaMA model detection."""
        assert model_type("meta-llama/Llama-2-7b-hf") == "llama"
        assert model_type("llama-7b") == "llama"
    
    def test_baichuan_detection(self):
        """Test Baichuan model detection."""
        assert model_type("baichuan-inc/Baichuan-7B") == "baichuan"
        assert model_type("Baichuan2-7B-Base") == "baichuan"
    
    def test_qwen_detection(self):
        """Test Qwen model detection."""
        assert model_type("Qwen/Qwen-1_8b") == "qwen"
        assert model_type("qwen-7b") == "qwen"
    
    def test_chatglm_detection(self):
        """Test ChatGLM model detection."""
        assert model_type("THUDM/chatglm2-6b") == "chatglm"
        assert model_type("chatglm-6b") == "chatglm"
    
    def test_internlm_detection(self):
        """Test InternLM model detection."""
        assert model_type("internlm/internlm-7b") == "internlm"
        assert model_type("internlm-chat-7b") == "internlm"
    
    def test_mistral_detection(self):
        """Test Mistral model detection."""
        assert model_type("mistralai/Mistral-7B-v0.1") == "mistral"
        assert model_type("mistral-7b") == "mistral"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestGetAttributes:
    """Test cases for get_attributes function."""
    
    def test_single_attribute(self):
        """Test getting a single attribute."""
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(10, 10)
        
        model = DummyModel()
        layer = get_attributes(model, "layer")
        assert isinstance(layer, nn.Linear)
    
    def test_nested_attributes(self):
        """Test getting nested attributes."""
        class InnerModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
        
        class OuterModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = InnerModule()
        
        model = OuterModule()
        linear = get_attributes(model, "inner.linear")
        assert isinstance(linear, nn.Linear)
    
    def test_deeply_nested_attributes(self):
        """Test getting deeply nested attributes."""
        class Level3(nn.Module):
            def __init__(self):
                super().__init__()
                self.value = 42
        
        class Level2(nn.Module):
            def __init__(self):
                super().__init__()
                self.level3 = Level3()
        
        class Level1(nn.Module):
            def __init__(self):
                super().__init__()
                self.level2 = Level2()
        
        model = Level1()
        value = get_attributes(model, "level2.level3.value")
        assert value == 42
    
    def test_invalid_attribute(self):
        """Test error handling for invalid attribute."""
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(10, 10)
        
        model = DummyModel()
        with pytest.raises(AttributeError):
            get_attributes(model, "nonexistent")
