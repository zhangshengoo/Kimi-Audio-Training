# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca and QwenLM/Qwen.

from dataclasses import dataclass, field
import json
import logging
import os
from typing import Dict, Optional, List

import torch
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, AutoTokenizer, AutoConfig
from transformers.integrations import deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from accelerate.utils import DistributedType
from huggingface_hub import snapshot_download
from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import GPTQConfig

from finetune_codes.model import KimiAudioModel
from finetune_codes.datasets import LazySupervisedDataset
from finetune_codes.lora_config import (
    LoraArguments, 
    get_optimized_lora_config, 
    merge_and_save, 
    print_lora_info, 
    save_lora_only,
    get_peft_state_maybe_zero_3
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="moonshotai/Kimi-Audio-7B-Instruct")
    model_path: str = field(
        default=None, metadata={"help": "Path to the pretrained model."}
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_ratio: float = field(
        default=0.05, metadata={"help": "Ratio of evaluation data."}
    )
    lazy_preprocess: bool = False

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    dataloader_pin_memory: bool = field(default=False)
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = field(default=False)

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)

def make_supervised_data_module(
    whisper_model, text_tokenizer, data_args, max_len, kimia_token_offset,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = LazySupervisedDataset
    rank0_print("Loading data...")

    with open(data_args.data_path, "r") as f:
        lines = f.readlines()
        all_data = [json.loads(line) for line in lines]

    if data_args.eval_ratio > 0:
        eval_data = all_data[:int(len(all_data) * data_args.eval_ratio)]
        train_data = all_data[int(len(all_data) * data_args.eval_ratio):]
        assert len(eval_data) > 0, "No evaluation data found"
        assert len(train_data) > 0, "No training data found"
    else:
        eval_data = None
        train_data = all_data

    train_dataset = dataset_cls(train_data, whisper_model=whisper_model, text_tokenizer=text_tokenizer, max_len=max_len, kimia_token_offset=kimia_token_offset)

    if eval_data:
        eval_dataset = dataset_cls(eval_data, whisper_model=whisper_model, text_tokenizer=text_tokenizer, max_len=max_len, kimia_token_offset=kimia_token_offset)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

def compute_loss(outputs, labels, num_items_in_batch=None):
    audio_logits, text_logits = outputs.logits
    audio_labels, text_labels, audio_loss_mask, text_loss_mask = labels
    assert audio_labels.shape[0] == 1, print("we only support micro batch size 1 for demo purpose")

    audio_loss = torch.nn.functional.cross_entropy(audio_logits.view(-1, audio_logits.shape[-1]), audio_labels.view(-1), reduction="none")
    text_loss = torch.nn.functional.cross_entropy(text_logits.view(-1, text_logits.shape[-1]), text_labels.view(-1), reduction="none")

    audio_loss = (audio_loss * audio_loss_mask.view(-1)).sum() / (audio_loss_mask.view(-1).sum() + 1e-4)
    text_loss = (text_loss * text_loss_mask.view(-1)).sum() / (text_loss_mask.view(-1).sum() + 1e-4)
    loss = audio_loss + text_loss
    return loss

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1))==1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logger.warning(
                "FSDP and ZeRO3 are not supported for QLoRA."
            )

    model_load_kwargs = {
        'low_cpu_mem_usage': not deepspeed.is_deepspeed_zero3_enabled(),
    }

    logger.info(f"Loading kimi-audio main model")

    if os.path.exists(model_args.model_name_or_path):
        cache_path = model_args.model_name_or_path
    else:
        cache_path = snapshot_download(model_args.model_name_or_path)

    logger.info(f"Looking for resources in {cache_path}")
    if not os.path.exists(model_args.model_path):
        raise ValueError(f"Model path {model_args.model_path} does not exist")

    # Load model with LoRA configuration
    if training_args.use_lora:
        if lora_args.q_lora:
            model = KimiAudioModel.from_pretrained(
                model_args.model_path,
                device_map=None,
                quantization_config=GPTQConfig(
                    bits=4, disable_exllama=True
                ),
                **model_load_kwargs
            )
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        else:
            # ToDo: should use init_from_pretrained when lora ?
            model = KimiAudioModel.from_pretrained(
                model_args.model_path,
                device_map=None,
                **model_load_kwargs
            )


        if training_args.use_lora:
            if lora_args.q_lora:
                modules_to_save = None
            else:
                modules_to_save = None #["wte" "lm_head"]
        if (
            training_args.use_lora
            and not lora_args.q_lora
            and deepspeed.is_deepspeed_zero3_enabled()
            and modules_to_save is not None
        ):
            raise RuntimeError("Deepspeed ZeRO3 is not supported for LoRA when using modules_to_save")

        # Configure LoRA
        lora_config = get_optimized_lora_config(lora_args)
        
        model = get_peft_model(model, lora_config)
        print_lora_info(lora_config, model)

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
    else:
        model = KimiAudioModel.from_pretrained(
            model_args.model_path,
            device_map=None,
            **model_load_kwargs
        )

    text_tokenizer = AutoTokenizer.from_pretrained(
        cache_path, trust_remote_code=True
    )

    # Load data
    data_module = make_supervised_data_module(
        whisper_model=model.whisper_model if hasattr(model, 'whisper_model') else model.base_model.whisper_model,
        text_tokenizer=text_tokenizer,
        data_args=data_args,
        max_len=training_args.model_max_length,
        kimia_token_offset=model.config.kimia_token_offset if hasattr(model, 'config') else model.base_model.config.kimia_token_offset
    )

    # Start trainer
    trainer = Trainer(
        model=model, args=training_args, 
        compute_loss_func=compute_loss,
        data_collator=data_module["train_dataset"].collate_fn,
        **data_module
    )

    trainer.train()
    trainer.save_state()

    # Save model
    if training_args.use_lora:
        # Save LoRA weights
        lora_output_dir = os.path.join(training_args.output_dir, "lora_weights")
        save_lora_only(model, lora_args, lora_output_dir)
        
        # Optionally merge and save the full model
        if lora_args.merge_lora_after_training:
            merged_output_dir = os.path.join(training_args.output_dir, "merged_model")
            merge_and_save(model, merged_output_dir, text_tokenizer)
    else:
        safe_save_model_for_hf_trainer(
            trainer=trainer,
            output_dir=training_args.output_dir,
            bias="none"
        )

if __name__ == "__main__":
    train()