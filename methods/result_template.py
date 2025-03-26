result = {   
    "origin_data": "any", # Raw data for each interpretation method, used for subsequent interaction. The specific meaning should be explained in the README of each interpretation method.
    "image": [{"image_name": "xxx", "image_path": "xxx"}, {"image_name": "xxx", "image_path": "xxx"}], # Name and path of each image
    "table": [{"table_name": "xxx1", "table_list": [{"a1": 1, "b1": 2}, {"a2": 3, "b2": 4}]}, 
                {"table_name": "xxx2", "table_list": [{"a2": 1, "b2": 2}, {"a2": 3, "b2": 4}]}] # Name and content of each table, where the content is organized as List[Dict]. Each Dict represents a row with its corresponding values. 
}