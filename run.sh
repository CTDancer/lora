export WANDB_API_KEY=b7b6ecceb6854bd12f58809f18264f979509d13b
export CUDA_HOME=/usr/local/cuda
export CUDA_VISIBLE_DEVICES=3
python finetune.py \
    --base_model 'huggyllama/llama-7b' \
    --data_path 'alpaca_data_cleaned_archive.json' \
    --output_dir './lora-alpaca-buffer' \
    --save_interval 1 \
    --save_dir '/shared/dqwang/scratch/tongchen/lora-buffer/' \
    --val_set_size 200 