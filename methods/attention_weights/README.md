# Output Explanation
```json
{
        "tokens": "Tokenized input text", 
        "attention_weight": "3D array: [num_layers][num_heads][num_tokens×num_tokens] attention matrices",
        "min_var_id": "Selected attention heads with lowest variance in attention distribution",
        "imgs": [
            {
                "image_name": "Attention head identifier (Layer-Head combination)",
                "image_path": "Temporary PNG file path for visualization"
            }
        ]
}
```