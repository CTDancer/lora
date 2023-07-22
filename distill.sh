export CUDA_HOME=/usr/local/cuda
export CUDA_VISIBLE_DEVICES=1
python distill.py \
    --base_model 'huggyllama/llama-7b' \
    --data_path 'distilled_dataset.json' \
    --val_path 'distilled_dataset.json' \
    --output_dir './lora-alpaca-distill' \
    --expert_dir '/shared/dqwang/scratch/tongchen/lora-official-2/' \
    --cutoff_len=512 \
    --group_by_length \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    --micro_batch_size=8 \
    --lr_teacher=1e-3 \
    --lr_text=0.1 \
    --lr_lr=1e-5 \
    --syn_steps=1
