import os
import sys
from typing import List
import logging
import fire
import torch
import transformers
from datasets import load_dataset, load_metric
import torch.nn as nn
import evaluate
import numpy as np
from utils.get_model import get_model

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter
from utils.save_callback import SavePeftModelCallback

def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    save_interval: int = 1,
    save_dir: str = "",
    num_experts: int = 1,
):
    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,  # Set the desired log level
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/{}.log'.format(wandb_run_name)),  # Save logs to a file
            logging.StreamHandler()  # Print logs to the console
        ]
    )

    if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    
    for it in range(0, num_experts):

        # get the model for training
        model, gradient_accumulation_steps, ddp = get_model(base_model=base_model, batch_size=batch_size, micro_batch_size=micro_batch_size,
                                                            lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_target_modules=lora_target_modules)

        # use wandb
        use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
        )
        # Only overwrite environ if wandb param passed
        if len(wandb_project) > 0:
            os.environ["WANDB_PROJECT"] = wandb_project
        if len(wandb_watch) > 0:
            os.environ["WANDB_WATCH"] = wandb_watch
        if len(wandb_log_model) > 0:
            os.environ["WANDB_LOG_MODEL"] = wandb_log_model
        wandb_run_name += "_it={}".format(it)


        # define tokenizer
        prompter = Prompter(prompt_template_name)
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        tokenizer.padding_side = "left"  # Allow batched inference

        def tokenize(prompt, add_eos_token=True):
            # there's probably a way to do this with the tokenizer settings
            # but again, gotta move fast
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
                ] * user_prompt_len + tokenized_full_prompt["labels"][
                    user_prompt_len:
                ]  # could be sped up, probably
            return tokenized_full_prompt


        # get data
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            data = load_dataset("json", data_files=data_path)
        else:
            data = load_dataset(data_path)

        if val_set_size > 0:
            train_val = data["train"].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            train_data = (
                train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            val_data = (
                train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
        else:
            train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
            val_data = None


        if resume_from_checkpoint:
            # Check the available weights and load them
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "pytorch_model.bin"
            )  # Full checkpoint
            if not os.path.exists(checkpoint_name):
                checkpoint_name = os.path.join(
                    resume_from_checkpoint, "adapter_model.bin"
                )  # only LoRA model - LoRA config above has to fit
                resume_from_checkpoint = (
                    False  # So the trainer won't try loading its state
                )
            # The two files above have a different name depending on how they were saved, but are actually the same.
            if os.path.exists(checkpoint_name):
                print(f"Restarting from {checkpoint_name}")
                adapters_weights = torch.load(checkpoint_name)
                set_peft_model_state_dict(model, adapters_weights)
            else:
                print(f"Checkpoint {checkpoint_name} not found")

        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.


        # define evaluation metric
        # metric = load_metric('bleu')
        
        # def compute_metrics(eval_pred):
        #     logits, labels = eval_pred
        #     predictions = np.argmax(logits, axis=-1)
        #     predictions = predictions.tolist()
        #     for i in range(len(predictions)):
        #         predictions[i] = [x for x in predictions[i] if (x != 0 and x != 30965)]
        #     preds = [tokenizer.convert_tokens_to_string(encoded_text) for encoded_text in predictions]
        #     preds = [sentence.split() for sentence in preds]
        #     labels = labels.tolist()
        #     for i in range(len(labels)):
        #         labels[i] = [x for x in labels[i] if x != -100]
        #     labs = [tokenizer.convert_tokens_to_string(encoded_text) for encoded_text in labels]
        #     labs = [[sentence.split()] for sentence in labs]
        #     return metric.compute(predictions=preds, references=labs)

        metric = load_metric('accuracy')
        
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            predictions = predictions.tolist()
            # remove values 0 and 30965 from each batch of predictions
            predictions = [[x for x in batch] for batch in predictions]
            labels = labels.tolist()
            # remove values -100 from each batch of labels
            labels = [[x for x in batch] for batch in labels]
            # compute accuracy for each batch and average the results
            accuracies = [metric.compute(predictions=p, references=r)['accuracy'] for p, r in zip(predictions, labels)]
            return {'accuracy': np.mean(accuracies)}

        # define trainer
        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            compute_metrics=compute_metrics,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=100,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
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
                report_to="wandb" if use_wandb else None,
                run_name=wandb_run_name if use_wandb else None,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
            callbacks=[SavePeftModelCallback],
        )
        model.config.use_cache = False

        timestamps = []

        old_state_dict = model.state_dict

        timestamps.append({name: param for name, param in model.named_parameters()})  

        # model.state_dict = (
        #     lambda self, *_, **__: get_peft_model_state_dict(
        #         self, old_state_dict()
        #     )
        # ).__get__(model, type(model))

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)


        trajectories = []

        _, timestamps = trainer.train(resume_from_checkpoint=resume_from_checkpoint, timestamps=timestamps)

        print("timestamps length: ", len(timestamps))

        trajectories.append(timestamps)
        if len(trajectories) == save_interval:
                n = 0
                while os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))):
                    n += 1
                print("Saving {}".format(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))))
                torch.save(trajectories, os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))
                trajectories = []

        model.save_pretrained(output_dir)

        trainer.evaluate()

        print(
            "\n If there's a warning about missing keys above, please disregard :)"
        )


if __name__ == "__main__":
    fire.Fire(train)
