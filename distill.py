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
from torch.utils.data import Dataset, TensorDataset
import logging
import wandb
import torch.nn.functional as F
from utils.reparam_module import ReparamModule

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

# define the dataset used for creating dataset from `inputs`
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def main(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "./distilled_dataset",
    val_path: str = "./alpaca_data_cleaned_archive",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    cutoff_len: int = 256,
    val_set_size: int = 200,
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
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    # distillation hyperparameters
    Iteration: int = 2000,
    load_all: bool = True,
    max_files: int = None,
    max_experts: int = None,
    expert_dir: str = "",
    max_start_epoch: int = 1,
    expert_epochs: int = 1,
    syn_steps: int = 5,
    lr_teacher: float = 1e-3,
    lr_text: float = 1e2,
    lr_lr: float = 1e-5,
    eval_it: int = 200,
    epoch_eval_train: int = 20
): 
    # Configure the root logger
    # logging.basicConfig(
    #     level=logging.INFO,  # Set the desired log level
    #     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    #     handlers=[
    #         logging.FileHandler('logs/distill.log'),  # Save logs to a file
    #         logging.StreamHandler()  # Print logs to the console
    #     ]
    # )

    # wandb.init(sync_tensorboard=False,
    #             entity='tongchen',
    #             project="alpaca-lora-distill",
    #             name="size=100-lr_teacher={}-lr_text={}-lr_lr={}-syn_steps={}".format(lr_teacher, lr_text, lr_lr, syn_steps)
    #         #    name='test'
    #            )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    distributed = torch.cuda.device_count() > 1

    # get the model for training
    model, gradient_accumulation_steps, ddp = get_model(base_model=base_model, batch_size=batch_size, micro_batch_size=micro_batch_size,
                                                        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_target_modules=lora_target_modules)
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.


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
        distilled_data = load_dataset("json", data_files=data_path)
    else:
        distilled_data = load_dataset(data_path)

    if val_path.endswith(".json") or val_path.endswith(".jsonl"):
        val_data = load_dataset("json", data_files=val_path)
    else:
        val_data = load_dataset(val_path)

    train_data = distilled_data["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = val_data["train"].shuffle().map(generate_and_tokenize_prompt)

    if val_set_size < len(val_data):
        val_data = val_data[:val_set_size]


    # define the evaluation metrics
    metric = load_metric('bleu')

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        predictions = predictions.tolist()
        for i in range(len(predictions)):
            predictions[i] = [x for x in predictions[i] if (x != 0 and x != 30965)]
        preds = [tokenizer.convert_tokens_to_string(encoded_text) for encoded_text in predictions]
        preds = [sentence.split() for sentence in preds]
        labels = labels.tolist()
        for i in range(len(labels)):
            labels[i] = [x for x in labels[i] if x != -100]
        labs = [tokenizer.convert_tokens_to_string(encoded_text) for encoded_text in labels]
        labs = [[sentence.split()] for sentence in labs]
        return metric.compute(predictions=preds, references=labs)


    syn_lr = torch.tensor(lr_teacher).to(device)
    syn_lr = syn_lr.detach().to(device).requires_grad_(True)

    temp_trainer = transformers.Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=None,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=0,
                fp16=True,
                logging_steps=1,
                optim="adamw_torch",
                output_dir=output_dir,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=group_by_length,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
            callbacks=[SavePeftModelCallback],
        )
    # get the inputs for trainer
    dataloader = temp_trainer.get_train_dataloader(batch_size=2)
    keys = []
    for batch in dataloader:
        input = temp_trainer._prepare_inputs(batch)
        keys = [k for k in input]
        break

    syn_text = [[] for _ in range(len(keys))]
    masks = []
    for batch in dataloader:
        input = temp_trainer._prepare_inputs(batch)
        for k, v in input.items():
            if k == 'input_ids':
                one_hot_v = F.one_hot(v, num_classes=32000)
                v = F.gumbel_softmax(one_hot_v.float(), tau=1.0, hard=False, dim=-1).requires_grad_(True)
                syn_text[0].append(v)
            elif k == 'labels':
                mask = (v == -100).float()
                masks.append(mask)
                v[v == -100] = 0
                one_hot_v = F.one_hot(v, num_classes=32000)
                gumbel_v = F.gumbel_softmax(one_hot_v.float(), tau=1.0, hard=False, dim=-1)
                syn_text[2].append(gumbel_v.requires_grad_(True))
            else:
                for i, key in enumerate(keys):
                    if k == key:
                        syn_text[i].append(v.float().requires_grad_(True))
                        break

    # define the optimizers
    optimizer_list = [None] * (len(keys))
    for i in range(len(optimizer_list)):
        optimizer_list[i] = torch.optim.SGD(syn_text[i], lr=lr_text, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=lr_lr, momentum=0.5)

    eval_it_pool = np.arange(0, Iteration + 1, eval_it).tolist()[1:]

    del temp_trainer
    del model
    torch.cuda.empty_cache()

    if load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)), map_location='cuda:0')
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
        buffer = torch.load(expert_files[file_idx], map_location='cuda:0')
        if max_experts is not None:
            buffer = buffer[:max_experts]
        random.shuffle(buffer)

    best_metric = 0

    for it in range(0, Iteration+1):
        # wandb.log({"Progress": it}, step=it)
        save_this_it = False

        model, gradient_accumulation_steps, ddp = get_model(base_model=base_model, batch_size=batch_size, micro_batch_size=micro_batch_size,
                                                            lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_target_modules=lora_target_modules)
        
        expert_trajectory = buffer
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
                    buffer = torch.load(expert_files[file_idx], map_location='cuda:0')
                if max_experts is not None:
                    buffer = buffer[:max_experts]
                random.shuffle(buffer)
        
        # # get dataset for evaluation or saving
        # if (it in eval_it_pool) or save_this_it or (it % 1000 == 0):
        #     batches = []
        #     for i in range(len(syn_text)):
        #         batch = {}
        #         for j in range(len(keys)):
        #             batch.update({keys[j]: torch.round(syn_text[i][:,:,j]).detach().long()})
        #         batches.append(batch)
            
        #     dataset = CustomDataset(batches)
        #     metadata = {
        #         'features': ['input_ids', 'attention_mask', 'labels'],
        #         'num_rows': len(dataset)
        #     }
        #     eval_dataset = {'Dataset': dataset, 'metadata': metadata}

        # if it in eval_it_pool:
        #     lr = syn_lr.item()

        #     # get model for evaluation
        #     eval_model, gradient_accumulation_steps, ddp = get_model(base_model=base_model, batch_size=batch_size, micro_batch_size=micro_batch_size,
        #                                                 lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_target_modules=lora_target_modules)

        #     eval_trainer = transformers.Trainer(
        #         model=eval_model,
        #         train_dataset=eval_dataset,
        #         eval_dataset=val_data,
        #         compute_metrics=compute_metrics,
        #         args=transformers.TrainingArguments(
        #             per_device_train_batch_size=micro_batch_size,
        #             gradient_accumulation_steps=gradient_accumulation_steps,
        #             warmup_steps=100,
        #             num_train_epochs=epoch_eval_train,
        #             learning_rate=float(lr),
        #             fp16=True,
        #             logging_steps=10,
        #             optim="adamw_torch",
        #             evaluation_strategy="steps" if val_set_size > 0 else "no",
        #             save_strategy="steps",
        #             eval_steps=200 if val_set_size > 0 else None,
        #             save_steps=200,
        #             output_dir=output_dir,
        #             save_total_limit=3,
        #             load_best_model_at_end=True if val_set_size > 0 else False,
        #             ddp_find_unused_parameters=False if ddp else None,
        #             group_by_length=group_by_length,
        #         ),
        #         data_collator=transformers.DataCollatorForSeq2Seq(
        #             tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        #         ),
        #         callbacks=[SavePeftModelCallback],
        #     )
        #     eval_model.config.use_cache = False

        #     old_state_dict = eval_model.state_dict
        #     eval_model.state_dict = (
        #         lambda self, *_, **__: get_peft_model_state_dict(
        #             self, old_state_dict()
        #         )
        #     ).__get__(eval_model, type(eval_model))

        #     if torch.__version__ >= "2" and sys.platform != "win32":
        #         eval_model = torch.compile(eval_model)

        #     eval_trainer.train()
        #     metric = eval_trainer.evaluate()

        #     print("metric: ", metric)

        #     if (metric['eval_bleu'] > best_metric):
        #         best_metric = metric
        #         save_this_it = True
            
        #     # wandb.log({'BLEU': metric['eval_bleu']}, step=it)

        # if (save_this_it or it % 1000 == 0):
        #     with torch.no_grad():
        #         text_save = eval_dataset
        #         save_dir = os.path.join(".", "logged_files")

        #         if not os.path.exists(save_dir):
        #             os.makedirs(save_dir)

        #         torch.save(text_save, os.path.join(save_dir, "text_{}.json".format(it)))

        #         if save_this_it:
        #             torch.save(text_save, os.path.join(save_dir, "text_best.json"))
        #             torch.save(syn_lr.item(), os.path.join(save_dir, "lr_best.pt"))


        # wandb.log({"Synthetic_LR": syn_lr.detach().cpu()}, step=it)
        
        target_params = expert_trajectory[-1]
        target_params = torch.cat([v.data.to(device).reshape(-1) for k, v in target_params.items() if 'lora_' in k or 'bias' in k], 0)

        # starting_params = torch.cat([v.data.to(device).reshape(-1) for k, v in model.named_parameters() if 'lora_' in k or 'bias' in k], 0)
        student_params = [torch.cat([v.data.to(device).reshape(-1) for k, v in model.named_parameters() if 'lora_' in k or 'bias' in k], 0).requires_grad_(True)]

        model = ReparamModule(model)

        if distributed:
            model = torch.nn.DataParallel(model)

        model.zero_grad()

        for epoch in range(0, syn_steps):

            syn_trainer = transformers.Trainer(
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
                    output_dir=output_dir,
                    ddp_find_unused_parameters=False if ddp else None,
                    group_by_length=group_by_length,
                ),
                data_collator=transformers.DataCollatorForSeq2Seq(
                    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
                ),
                callbacks=[SavePeftModelCallback],
            )

            if torch.__version__ >= "2" and sys.platform != "win32":
                model = torch.compile(model)

            index = random.randint(0, len(syn_text[0])-1)
            input_dict = {}
            for i in range(len(keys)):
                input_dict[keys[i]] = syn_text[i][index]

            model.train()

            prepared_inputs = syn_trainer._prepare_inputs(input_dict)

            if distributed:
                forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params = student_params[-1]

            with syn_trainer.compute_loss_context_manager():
                loss = syn_trainer.compute_loss(model, prepared_inputs, flat_param=forward_params)
                mask = masks[index]
                loss = (loss * (1 - mask)).sum() / (1 - mask).sum()
                print("loss: ", loss)

            grad = torch.autograd.grad(loss, student_params[-1], create_graph=True)[0]

            tmp = student_params[-1] - syn_lr * grad
            student_params.append(tmp)


        # use distilled dataset to train the model for syn_steps steps and get parameters along the way
        # student_params = None
        # student_params = syn_trainer.syn_train(inputs=syn_text, keys=keys, masks=masks, lr=syn_lr, student_params=student_params)

        # compute parameter distance loss and update distilled dataset
        param_loss = torch.tensor(0.0).to(device)
        param_dist = torch.tensor(0.0).to(device)

        # for k, v in target_params.items():
        #     if 'lora_' in k or 'bias' in k:
        #         param_loss += torch.dist(student_params[k], target_params[k])
        #         param_dist += torch.dist(starting_params[k], target_params[k])

        param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist += torch.nn.functional.mse_loss(student_params[0], target_params, reduction="sum")
        print("iter = %04d, param_loss = %.4f, param_dist = %.4f" % (it, param_loss, param_dist))
        param_loss /= param_dist

        grand_loss = param_loss

        for optimizer in optimizer_list:
            optimizer.zero_grad()
        optimizer_lr.zero_grad()

        grand_loss.backward()

        # Print out the gradients
        for i in range(len(syn_text)):
            print("The gradients of {}:".format(keys[i]))
            for j in range(len(syn_text[i])):
                print(syn_text[i][j].grad)

        print("Gradient of syn_lr:", syn_lr.grad)

        for optimizer in optimizer_list:
            optimizer.step()
        optimizer_lr.step()

        # wandb.log({"Grand_Loss": grand_loss.detach().cpu()})

        for _ in student_params:
            del _

        del syn_trainer
        del model
        torch.cuda.empty_cache()

        if it%10 == 0:
            print('iter = %04d, loss = %.4f' % (it, grand_loss.item()))

    # wandb.finish()

if __name__ == "__main__":
    fire.Fire(main)
