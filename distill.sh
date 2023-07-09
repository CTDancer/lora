export WANDB_API_KEY=b7b6ecceb6854bd12f58809f18264f979509d13b
export CUDA_HOME=/usr/local/cuda
export CUDA_VISIBLE_DEVICES=4
python distill.py \
    --base_model 'huggyllama/llama-7b' \
    --data_path 'distilled_dataset.json' \
    --val_path 'distilled_dataset.json' \
    --output_dir './lora-alpaca-buffer' \
    --expert_dir '/shared/dqwang/scratch/tongchen/lora-buffer/' \
    --val_set_size 200
