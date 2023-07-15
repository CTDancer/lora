import os
import fire
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset, load_metric
import numpy as np

from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from utils.get_model import get_model
from utils.prompter import Prompter
from utils.save_callback import SavePeftModelCallback

def main(
    base_model: str="",
    lora_model: str="",
    data_path: str="",
    val_set_size: int=2000,
    output_dir: str="",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    save_interval: int = 1,
    save_dir: str = "",
    num_experts: int = 1,
):

    assert base_model, "Please specify a value for the base_model"
    assert val_set_size > 0, "Please specify a length for the evaluation dataset"

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    prompter = Prompter(prompt_template_name)
    tokenizer.pad_token_id = 0  # Set padding token ID
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

        return tokenized_full_prompt

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    base_model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_model,
        device_map=device_map,
        torch_dtype=torch.float16,
    )

    # Merge weights
    for layer in lora_model.base_model.model.model.layers:
        layer.self_attn.q_proj.merge_weights = True
        layer.self_attn.v_proj.merge_weights = True

    lora_model.train(False)

    metric = load_metric('accuracy')

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        predictions = predictions.tolist()
        predictions = [[x for x in batch if x not in [0, 30965]] for batch in predictions]
        labels = labels.tolist()
        labels = [[x for x in batch if x != -100] for batch in labels]
        accuracies = [metric.compute(predictions=p, references=r)['accuracy'] for p, r in zip(predictions, labels)]
        return {'accuracy': np.mean(accuracies)}

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    train_val = data["train"].train_test_split(
        test_size=val_set_size, shuffle=True, seed=42
    )
    val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)

    evaluator = transformers.Trainer(
        model=lora_model,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
        args=transformers.TrainingArguments(
            warmup_steps=100,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[SavePeftModelCallback],
    )

    metric = evaluator.evaluate()
    print("metric: ", metric)

if __name__ == "__main__":
    fire.Fire(main)
