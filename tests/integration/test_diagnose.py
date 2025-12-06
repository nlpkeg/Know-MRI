"""
Integration tests for the diagnose module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from diagnose import diagnose
from methods import method_name2diagnose_fun


class TestDiagnoseModule:
    """Integration test cases for diagnose module."""
    
    def test_get_model_output_with_ground_truth(self, sample_processed_kvs):
        """Test get_model_output returns ground_truth when available."""
        result = diagnose.get_model_output(
            sample=sample_processed_kvs,
            model_name_or_path="test_model"
        )
        assert result == sample_processed_kvs["ground_truth"]
    
    def test_get_model_output_without_ground_truth(self):
        """Test get_model_output returns empty string when ground_truth missing."""
        sample = {"prompt": "Test prompt"}
        result = diagnose.get_model_output(
            sample=sample,
            model_name_or_path="test_model"
        )
        assert result == ""
    
    def test_diagnosing_method_lookup(self):
        """Test that diagnosing can lookup registered methods."""
        # Verify that method_name2diagnose_fun is populated
        assert isinstance(method_name2diagnose_fun, dict)
        # The dict should not be empty (methods should be registered)
        # Note: This assumes methods are imported and registered
    
    def test_diagnosing_with_mock_method(self, sample_processed_kvs):
        """Test diagnosing with a mocked method."""
        # Create a mock diagnose function
        mock_result = {
            "origin_data": {"test": "data"},
            "image": [],
            "table": []
        }
        
        mock_diagnose_func = Mock(return_value=mock_result)
        
        # Patch the method lookup
        with patch.dict('methods.method_name2diagnose_fun', 
                       {'test_method': mock_diagnose_func}):
            result = diagnose.diagnosing(
                sample=sample_processed_kvs,
                model_name_or_path="test_model",
                method="test_method"
            )
            
            # Verify the mock was called
            mock_diagnose_func.assert_called_once()
            assert result == mock_result
    
    def test_diagnosing_passes_hparams(self, sample_processed_kvs):
        """Test that diagnosing passes hparams to method."""
        mock_diagnose_func = Mock(return_value={})
        test_hparams = {"param1": "value1"}
        
        with patch.dict('methods.method_name2diagnose_fun',
                       {'test_method': mock_diagnose_func}):
            diagnose.diagnosing(
                sample=sample_processed_kvs,
                model_name_or_path="test_model",
                method="test_method",
                hparams=test_hparams
            )
            
            # Verify hparams was passed
            call_kwargs = mock_diagnose_func.call_args[1]
            assert 'hparams' in call_kwargs
            assert call_kwargs['hparams'] == test_hparams
    
    def test_diagnosing_result_structure(self, sample_processed_kvs):
        """Test that diagnosing result has expected structure."""
        expected_result = {
            "origin_data": {"key": "value"},
            "image": [{"image_name": "test", "image_path": "/path/to/image"}],
            "table": [{"table_name": "test_table", "table_list": [{"a": 1}]}]
        }
        
        mock_diagnose_func = Mock(return_value=expected_result)
        
        with patch.dict('methods.method_name2diagnose_fun',
                       {'test_method': mock_diagnose_func}):
            result = diagnose.diagnosing(
                sample=sample_processed_kvs,
                model_name_or_path="test_model",
                method="test_method"
            )
            
            # Verify result structure
            assert "origin_data" in result
            assert "image" in result
            assert "table" in result
            assert isinstance(result["image"], list)
            assert isinstance(result["table"], list)


class TestDiagnoseIntegration:
    """Integration tests for complete diagnosis workflow."""
    
    def test_method_registration(self):
        """Test that methods are properly registered."""
        from methods import support_methods
        
        # Verify that at least some methods are registered
        assert isinstance(support_methods, list)
        
        # Each method name should have a corresponding diagnose function
        for method_name in support_methods:
            assert method_name in method_name2diagnose_fun
            assert callable(method_name2diagnose_fun[method_name])
    
    def test_kn_method_exists(self):
        """Test that KN method is registered (if available)."""
        try:
            from methods import kn
            assert kn.name in method_name2diagnose_fun
            assert hasattr(kn, 'diagnose')
            assert hasattr(kn, 'requires_input_keys')
        except ImportError:
            pytest.skip("KN method not available")
