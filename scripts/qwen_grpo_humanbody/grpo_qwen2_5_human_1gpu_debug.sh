experient_name=grpo_human_body


CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file asset/config/deepspeed_zero2_1gpus.yaml \
    basic_trainer_submit.py \
    --dataset_name "/m2v_intern/wangqunzhong/research/asset/kwai_data/dataset" \
    --output_dir log/model_checkpoints/$experient_name \
    --remove_unused_columns False\
    --learning_rate 5e-6 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_iterations 8 \
    --num_generations 8 \
    --model_name_or_path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --report_to "wandb" \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 100 \
    --num_train_epochs 5 \
    --use_peft True \
    --lora_task_type "CAUSAL_LM" \
    --lora_r 64 \
    --lora_target_modules "q_proj" "v_proj"\
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --remove_unused_columns False \
    --log_completions True \
    --data_pipeline "qwen2.5-humanbody-grpo" \
    --data_select_ratio 0.05 \
    --cache_dir "/m2v_intern/wangqunzhong/research/asset/huggingface/model/Qwen/Qwen2.5-VL-7B-Chat" \
    --torch_dtype "bfloat16" \
    --debug_entry_point True \
    --data_source "video" \
    --do_train True \
    --bf16 True \
    --max_completion_length 256 \
    --fps 1. \
    --max_prompt_length 512

