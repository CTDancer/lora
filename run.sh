export WANDB_API_KEY=b7b6ecceb6854bd12f58809f18264f979509d13b
export CUDA_HOME=/usr/local/cuda
export CUDA_VISIBLE_DEVICES=3
python finetune.py \
    --val_set_size 200 \
    --learning_rate 1e-5 \
    --num_experts 1 \
    --save_interval 1 \
    --base_model 'huggyllama/llama-7b' \
    --data_path 'alpaca_data_cleaned_archive.json' \
    --output_dir './lora-alpaca-buffer' \
    --save_dir '/shared/dqwang/scratch/tongchen/lora-buffer/' \
    --wandb_project 'alpaca-lora-buffer' \
    --wandb_name 'lr=1e-5'