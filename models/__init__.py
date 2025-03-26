# The model name (must match Hugging Face's naming) is used to locate the corresponding JSON file for each method's hyperparameters.

gpt2 = "openai-community/gpt2-xl"
bert = "google-bert/bert-base-uncased"
gptj = "EleutherAI/gpt-j-6b"
llama = "meta-llama/Llama-2-7b-hf"
baichuan = "baichuan-inc/Baichuan-7B"
t5 = "google/flan-t5-large"
chatglm2 = "THUDM/chatglm2-6b"
internlm = "internlm/internlm-xcomposer2d5-ol-7b"
qwen = "Qwen/Qwen-1_8b"
mistral = "mistralai/Mistral-Nemo-Instruct-2407"

support_models = [
    llama, gpt2, bert, gptj, baichuan, t5, chatglm2, internlm, qwen, mistral
]