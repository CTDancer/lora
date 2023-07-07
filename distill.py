import os
import sys
from typing import List
import random
import fire
import torch
import transformers
from datasets import load_dataset, load_metric
import numpy as np
import torch.nn as nn
import evaluate

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter
from utils.get_model import get_model
from utils.save_callback import SavePeftModelCallback


def main(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "./distilled_dataset",
    val_path: str = "./alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
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
    # distillation hyperparameters
    Iteration: int = 2000,
    load_all: bool = False,
    max_files: int = None,
    max_experts: int = None,
    expert_dir: str = "",
    max_start_epoch: int = 1,
    expert_epochs: int = 2,
    syn_steps: int = 5,
    lr_teacher: float = 1e-3,
    lr_text: float = 1e2,
    lr_lr: float = 1e-5,
    eval_it: int = 200,
    epoch_eval_train: int = 500
): 

    model, gradient_accumulation_steps, ddp = get_model(base_model=base_model, batch_size=batch_size, micro_batch_size=micro_batch_size,
                                                        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_target_modules=lora_target_modules)

    prompter = Prompter(prompt_template_name)

    # Check if parameter passed or if set within environ
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


    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        distilled_data = load_dataset("json", data_files=data_path)
    else:
        distilled_data = load_dataset(data_path)

    if val_path.endswith(".json") or val_path.endswith(".jsonl"):
        val_data = load_dataset("json", data_files=val_path)
    else:
        val_data = load_dataset(val_path)

    train_data = distilled_data["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = val_data["train"].shuffle().map(generate_and_tokenize_prompt)

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    metric = load_metric('sacrebleu')

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        predictions = predictions.tolist()
        preds = [tokenizer.convert_tokens_to_string(encoded_text) for encoded_text in predictions]
        # print("preds: ", preds)
        labels = labels.tolist()
        labs = [tokenizer.convert_tokens_to_string(encoded_text) for encoded_text in labels]
        # print("labs: ", labs)
        return metric.compute(predictions=preds, references=labs)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    syn_lr = torch.tensor(lr_teacher).to(device)
    syn_lr = syn_lr.detach().to(device).requires_grad_(True)

    optimizer_text = torch.optim.SGD([train_data], lr=lr_text, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=lr_lr, momentum=0.5)
    optimizer_text.zero_grad()

    eval_it_pool = np.arange(0, Iteration + 1, eval_it).tolist()[1:]

    if load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)
        if max_files is not None:
            expert_files = expert_files[:max_files]
        print("loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        if max_experts is not None:
            buffer = buffer[:max_experts]
        random.shuffle(buffer)

    best_metric = 0

    for it in range(0, Iteration+1):
        save_this_it = False

        if load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                print("loading file {}".format(expert_files[file_idx]))
                if max_files != 1:
                    del buffer
                    buffer = torch.load(expert_files[file_idx])
                if max_experts is not None:
                    buffer = buffer[:max_experts]
                random.shuffle(buffer)
        
        if it in eval_it_pool:
            model, gradient_accumulation_steps, _ = get_model(base_model='decapoda-research/llama-7b-hf')

            lr = syn_lr.item()

            trainer = transformers.Trainer(
                model=model,
                train_dataset=train_data,
                eval_dataset=val_data,
                compute_metrics=compute_metrics,
                args=transformers.TrainingArguments(
                    per_device_train_batch_size=micro_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    warmup_steps=100,
                    num_train_epochs=epoch_eval_train,
                    learning_rate=float(lr),
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

            old_state_dict = model.state_dict
            model.state_dict = (
                lambda self, *_, **__: get_peft_model_state_dict(
                    self, old_state_dict()
                )
            ).__get__(model, type(model))

            if torch.__version__ >= "2" and sys.platform != "win32":
                model = torch.compile(model)

            trainer.train()
            metric = trainer.evaluate()

            print("metric: ", metric)

            if (metric > best_metric):
                best_metric = metric
                save_this_it = True

        if (save_this_it or it % 1000 == 0):
            with torch.no_grad():
                text_save = train_data.cuda()
                save_dir = os.path.join(".", "logged_files")

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                torch.save(text_save.cpu(), os.path.join(save_dir, "text_{}.json".format(it)))

                if save_this_it:
                    torch.save(text_save.cpu(), os.path.join(save_dir, "text_best.json"))
                    torch.save(syn_lr.item(), os.path.join(save_dir, "lr_best.pt"))

        start_epoch = np.random.randint(0, max_start_epoch)
        starting_params = expert_trajectory[start_epoch]

        target_params = expert_trajectory[start_epoch+expert_epochs]
        target_params = torch.cat([v.data.to(device).reshape(-1) for k, v in target_params], 0)

        student_params = [torch.cat([v.data.to(device).reshape(-1) for k, v in starting_params], 0).requires_grad_(True)]

        starting_params = torch.cat([v.data.to(device).reshape(-1) for k, v in starting_params], 0)

        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=None,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=0,
                num_train_epochs=syn_steps,
                learning_rate=syn_lr,
                fp16=True,
                logging_steps=1,
                optim="adamw_torch",
                evaluation_strategy="no",
                save_strategy="steps",
                eval_steps=None,
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

        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(model, type(model))

        for name, param in model.state_dict().items():
            if name in student_params[-1]:
                new_param = nn.Parameter(student_params[-1][name].data)
                model.state_dict()[name].copy_(new_param.data)

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        dataloader = trainer.get_train_dataloader()
        batch = next(iter(dataloader))
        inputs = trainer._prepare_inputs(batch)
        
        timestamps = []

        _, timestamps = trainer.train(inputs=inputs, timestamps=timestamps)

        param_loss = torch.tensor(0.0).to(device)
        param_dist = torch.tensor(0.0).to(device)

        for param_dict in timestamps:
            param = torch.cat([v.data.to(device).reshape(-1) for k, v in param_dict], 0).requires_grad_(True)
            student_params.append(param)

        param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

        param_loss /= param_dist

        grand_loss = param_loss

        optimizer_text.zero_grad()
        optimizer_lr.zero_grad()

        grand_loss.backward()

        optimizer_text.step()
        optimizer_lr.step()
        
        for _ in student_params:
            del _
        
        if it%10 == 0:
            print('iter = %04d, loss = %.4f' % (it, grand_loss.item()))