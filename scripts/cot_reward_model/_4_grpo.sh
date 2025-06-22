experient_name=grpo_human_body_qwen

CUDA_VISIBLE_DEVICES=0  RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=localhost MASTER_PORT=29500 python agent_grpo_trainer_submit.py \
    --dataset_name "/m2v_intern/wangqunzhong/research/asset/kwai_data/dataset" \
    --output_dir log/model_checkpoints/$experient_name \
    --remove_unused_columns False\
    --learning_rate 5e-6 \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 16 \
    --num_iterations 8 \
    --num_generations 48 \
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
    --data_select_ratio 0.1 \
    --cache_dir "/m2v_intern/wangqunzhong/research/asset/huggingface/model/Qwen/Qwen2.5-VL-7B-Chat" \
    --torch_dtype "bfloat16" \
    --debug_entry_point False \
    --data_source "video" \
    --do_train True \
    --bf16 True \
    --max_completion_length 1024 \
    --fps 8. \
    --max_prompt_length 2048 \
    --run_name $experient_name \
    --use_vllm True \
    --vllm_mode colocate \
    --vllm_tensor_parallel_size 1


