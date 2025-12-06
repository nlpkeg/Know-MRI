"""
Unit tests for methods/result_template.py.
"""
import pytest
from methods.result_template import result


class TestResultTemplate:
    """Test cases for result template structure."""
    
    def test_result_template_exists(self):
        """Test that result template dictionary exists."""
        assert result is not None
        assert isinstance(result, dict)
    
    def test_origin_data_key(self):
        """Test that origin_data key exists in template."""
        assert "origin_data" in result
    
    def test_image_key(self):
        """Test that image key exists and has correct structure."""
        assert "image" in result
        assert isinstance(result["image"], list)
        
        # Test image structure
        if len(result["image"]) > 0:
            image_item = result["image"][0]
            assert isinstance(image_item, dict)
            assert "image_name" in image_item
            assert "image_path" in image_item
    
    def test_table_key(self):
        """Test that table key exists and has correct structure."""
        assert "table" in result
        assert isinstance(result["table"], list)
        
        # Test table structure
        if len(result["table"]) > 0:
            table_item = result["table"][0]
            assert isinstance(table_item, dict)
            assert "table_name" in table_item
            assert "table_list" in table_item
            assert isinstance(table_item["table_list"], list)
    
    def test_required_keys(self):
        """Test that all required keys are present."""
        required_keys = ["origin_data", "image", "table"]
        for key in required_keys:
            assert key in result, f"Required key '{key}' missing from result template"
    
    def test_image_list_structure(self):
        """Test image list has proper structure for all items."""
        for item in result["image"]:
            assert isinstance(item, dict)
            assert "image_name" in item
            assert "image_path" in item
            assert isinstance(item["image_name"], str)
            assert isinstance(item["image_path"], str)
    
    def test_table_list_structure(self):
        """Test table list has proper structure for all items."""
        for table in result["table"]:
            assert isinstance(table, dict)
            assert "table_name" in table
            assert "table_list" in table
            assert isinstance(table["table_name"], str)
            assert isinstance(table["table_list"], list)
            
            # Check table_list items are dictionaries
            for row in table["table_list"]:
                assert isinstance(row, dict)


class TestResultTemplateUsage:
    """Test cases for using the result template."""
    
    def test_create_valid_result(self):
        """Test creating a valid result following the template."""
        new_result = {
            "origin_data": {"sample": "data"},
            "image": [
                {"image_name": "test_image", "image_path": "/path/to/image.png"}
            ],
            "table": [
                {
                    "table_name": "results",
                    "table_list": [
                        {"metric": "accuracy", "value": 0.95},
                        {"metric": "precision", "value": 0.92}
                    ]
                }
            ]
        }
        
        # Verify structure matches template
        assert set(new_result.keys()) == set(result.keys())
        assert isinstance(new_result["origin_data"], (dict, str, list))
        assert isinstance(new_result["image"], list)
        assert isinstance(new_result["table"], list)
    
    def test_empty_result(self):
        """Test creating a valid empty result."""
        empty_result = {
            "origin_data": None,
            "image": [],
            "table": []
        }
        
        # Should have all required keys
        assert "origin_data" in empty_result
        assert "image" in empty_result
        assert "table" in empty_result
    
    def test_multiple_images(self):
        """Test result with multiple images."""
        multi_image_result = {
            "origin_data": {},
            "image": [
                {"image_name": "img1", "image_path": "/path1.png"},
                {"image_name": "img2", "image_path": "/path2.png"},
                {"image_name": "img3", "image_path": "/path3.png"}
            ],
            "table": []
        }
        
        assert len(multi_image_result["image"]) == 3
        for img in multi_image_result["image"]:
            assert "image_name" in img
            assert "image_path" in img
    
    def test_multiple_tables(self):
        """Test result with multiple tables."""
        multi_table_result = {
            "origin_data": {},
            "image": [],
            "table": [
                {"table_name": "table1", "table_list": [{"a": 1}]},
                {"table_name": "table2", "table_list": [{"b": 2}]}
            ]
        }
        
        assert len(multi_table_result["table"]) == 2
        for table in multi_table_result["table"]:
            assert "table_name" in table
            assert "table_list" in table
