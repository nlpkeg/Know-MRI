# Output Explanation
```json
{
    "origin_data": {
        "model_output": "Model's prediction based on the prompt",
        "prob": "Confidence score of the model's prediction",
        "tokens": "Tokenized prompt as a list",
        "subject_range": "Subject span in prompt tokens (left-closed, right-open interval)",
        "Restoring_state_score": "np.array of shape len(tokens)*num_layer indicating each token's hidden state influence on model prediction per layer",
        "Restoring_MLP_score": "Same as Restoring_state_score",
        "Restoring_Attn_score": "Same as Restoring_state_score"
    }
}
```