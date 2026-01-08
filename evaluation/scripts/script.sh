CUDA_VISIBLE_DEVICES=0,1,2 python EvaluationWithEasyedit.py \
    --num_samples 500 \
    --data_type Zhujiu-MCScript \
    --model_path "./evaluation/models/gpt-j-6b" \
    --cache_dir "./evaluation/STATS_DIR/"