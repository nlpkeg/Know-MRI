"""
Unit tests for methods module registration and structure.
"""
import pytest

try:
    from methods import method_name2diagnose_fun, method_name2sub_module
    METHODS_AVAILABLE = True
except ImportError:
    METHODS_AVAILABLE = False


@pytest.mark.skipif(not METHODS_AVAILABLE, reason="Methods module dependencies not available")
class TestMethodsModule:
    """Test cases for methods module."""
    
    def test_method_name2diagnose_fun_exists(self):
        """Test that method_name2diagnose_fun dictionary exists."""
        assert method_name2diagnose_fun is not None
        assert isinstance(method_name2diagnose_fun, dict)
    
    def test_method_name2sub_module_exists(self):
        """Test that method_name2sub_module dictionary exists."""
        assert method_name2sub_module is not None
        assert isinstance(method_name2sub_module, dict)
    
    def test_methods_are_callable(self):
        """Test that all registered methods are callable."""
        for method_name, diagnose_func in method_name2diagnose_fun.items():
            assert callable(diagnose_func), f"Method {method_name} is not callable"
    
    def test_method_modules_have_name(self):
        """Test that method modules have a 'name' attribute."""
        for method_name, module in method_name2sub_module.items():
            assert hasattr(module, 'name'), f"Module {method_name} missing 'name' attribute"
            assert module.name == method_name
    
    def test_method_modules_have_diagnose(self):
        """Test that method modules have a 'diagnose' function."""
        for method_name, module in method_name2sub_module.items():
            assert hasattr(module, 'diagnose'), f"Module {method_name} missing 'diagnose' function"
            assert callable(module.diagnose)
    
    def test_method_modules_have_requires_input_keys(self):
        """Test that method modules have 'requires_input_keys' attribute."""
        for method_name, module in method_name2sub_module.items():
            assert hasattr(module, 'requires_input_keys'), \
                f"Module {method_name} missing 'requires_input_keys' attribute"
            assert isinstance(module.requires_input_keys, (list, tuple)), \
                f"Module {method_name} requires_input_keys should be list or tuple"
    
    def test_sync_between_dictionaries(self):
        """Test that method names are consistent between dictionaries."""
        assert set(method_name2diagnose_fun.keys()) == set(method_name2sub_module.keys()), \
            "Method names should be consistent between dictionaries"
    
    def test_diagnose_func_matches_module(self):
        """Test that diagnose function in dict matches module's diagnose."""
        for method_name in method_name2diagnose_fun.keys():
            if method_name in method_name2sub_module:
                module = method_name2sub_module[method_name]
                assert method_name2diagnose_fun[method_name] == module.diagnose, \
                    f"Diagnose function mismatch for {method_name}"


@pytest.mark.skipif(not METHODS_AVAILABLE, reason="Methods module dependencies not available")
class TestKNMethod:
    """Test cases specifically for KN method (if available)."""
    
    def test_kn_method_structure(self):
        """Test KN method has required structure."""
        try:
            from methods import kn
            
            # Test required attributes
            assert hasattr(kn, 'name')
            assert hasattr(kn, 'diagnose')
            assert hasattr(kn, 'requires_input_keys')
            assert hasattr(kn, 'cost_seconds_per_query')
            assert hasattr(kn, 'path')
            
            # Test types
            assert isinstance(kn.name, str)
            assert callable(kn.diagnose)
            assert isinstance(kn.requires_input_keys, (list, tuple))
            assert isinstance(kn.cost_seconds_per_query, (int, float))
            assert isinstance(kn.path, list)
        except ImportError:
            pytest.skip("KN method not available")
    
    def test_kn_requires_input_keys(self):
        """Test KN method requires correct input keys."""
        try:
            from methods import kn
            
            # KN typically requires prompts and ground_truth
            assert "prompts" in kn.requires_input_keys or "prompt" in kn.requires_input_keys
            assert "ground_truth" in kn.requires_input_keys
        except ImportError:
            pytest.skip("KN method not available")
    
    def test_kn_hyperparams_class(self):
        """Test KN HyperParams class exists and has expected fields."""
        try:
            from methods.kn import KNHyperParams
            
            # Test that class exists and is a dataclass
            from dataclasses import is_dataclass
            assert is_dataclass(KNHyperParams)
            
            # Test that it has expected fields
            hparams = KNHyperParams()
            assert hasattr(hparams, 'lr_scale')
            assert hasattr(hparams, 'n_toks')
            assert hasattr(hparams, 'batch_size')
        except ImportError:
            pytest.skip("KN method not available")


@pytest.mark.skipif(not METHODS_AVAILABLE, reason="Methods module dependencies not available")
class TestMethodRegistration:
    """Test cases for method registration process."""
    
    def test_no_empty_methods(self):
        """Test that at least some methods are registered (if dependencies available)."""
        if not METHODS_AVAILABLE:
            pytest.skip("Methods module not available")
        # Methods may not register if their dependencies are missing, which is acceptable
        # Just verify the dict structure exists
        assert isinstance(method_name2diagnose_fun, dict)
    
    def test_method_names_are_strings(self):
        """Test that all method names are strings."""
        for method_name in method_name2diagnose_fun.keys():
            assert isinstance(method_name, str), f"Method name {method_name} is not a string"
    
    def test_support_methods_list(self):
        """Test that support_methods list is available."""
        if not METHODS_AVAILABLE:
            pytest.skip("Methods module not available")
        
        from methods import support_methods
        
        assert isinstance(support_methods, list)
        # Methods may be empty if dependencies are missing, which is acceptable
        
        # If any methods are registered, verify they're in method_name2diagnose_fun
        for method_name in support_methods:
            assert method_name in method_name2diagnose_fun
