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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, TaskType
from transformers import GPTQConfig

from finetune_codes.model import KimiAudioModel
from finetune_codes.datasets import LazySupervisedDataset

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
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    # 新增：是否在训练结束后合并LoRA权重
    merge_lora_after_training: bool = field(
        default=False, metadata={"help": "Merge LoRA weights with base model after training"}
    )

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

def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias.items():
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

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

def merge_and_save(peft_model, output_dir: str, tokenizer=None):
    """Merge LoRA weights and save the model"""
    logger.info(f"Merging LoRA weights and saving to {output_dir}")
    assert isinstance(peft_model, PeftModel), "model must be a PeftModel when merging"

    # Merge LoRA weights
    merged_model = peft_model.merge_and_unload()
    
    # Save the merged model
    merged_model.save_pretrained(output_dir)
    
    # Save tokenizer if available
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    
    logger.info("Model merged and saved successfully")

def save_lora_only(peft_model, lora_args, output_dir: str):
    """Save only LoRA weights"""
    logger.info(f"Saving LoRA weights to {output_dir}")
    assert isinstance(peft_model, PeftModel), "model must be a PeftModel when saving"
    peft_model.save_pretrained(output_dir)
    
    # Save LoRA config
    lora_config_path = os.path.join(output_dir, "lora_config.json")
    with open(lora_config_path, 'w') as f:
        json.dump(vars(lora_args), f, indent=2)

def print_lora_info(lora_args, peft_model):
    """Print LoRA configuration and statistics"""
    logger.info("="*50)
    logger.info("LoRA Configuration:")
    logger.info(f"  Rank: {lora_args.lora_r}")
    logger.info(f"  Alpha: {lora_args.lora_alpha}")
    logger.info(f"  Dropout: {lora_args.lora_dropout}")
    logger.info(f"  Target modules: {lora_args.lora_target_modules}")
    
    # Print trainable parameters
    peft_model.print_trainable_parameters()
    
    # Detailed parameter statistics
    total_params = 0
    trainable_params = 0
    for name, param in peft_model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            logger.debug(f"  Trainable: {name} - {param.shape}")
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable ratio: {trainable_params/total_params:.2%}")
    logger.info("="*50)

def get_optimized_lora_config(
    lora_args,
    target_audio_modules=True,
    target_text_modules=True,
    target_mimo_modules=True,
):
    """
    Get optimized LoRA configuration for KimiAudio
    
    Args:
        lora_args: LoRA arguments containing r, alpha, dropout, etc.
        target_audio_modules: Whether to apply LoRA to audio-related modules
        target_text_modules: Whether to apply LoRA to text-related modules
        target_mimo_modules: Whether to apply LoRA to MIMO layers
    """
    target_modules = []
    
    # 如果用户指定了target_modules，优先使用用户的配置
    if lora_args.lora_target_modules:
        # 扩展用户指定的模块到所有层
        for module_suffix in lora_args.lora_target_modules:
            # Base transformer layer modules
            if target_text_modules:
                for i in range(28):  # Based on model config
                    target_modules.append(f"model.layers.{i}.self_attn.{module_suffix}")
                    if module_suffix in ["gate_proj", "up_proj", "down_proj"]:
                        target_modules.append(f"model.layers.{i}.mlp.{module_suffix}")
            
            # MIMO layers
            if target_mimo_modules:
                for i in range(8):  # Based on config.kimia_mimo_layers
                    target_modules.append(f"model.mimo_layers.{i}.self_attn.{module_suffix}")
                    if module_suffix in ["gate_proj", "up_proj", "down_proj"]:
                        target_modules.append(f"model.mimo_layers.{i}.mlp.{module_suffix}")
    else:
        # 使用默认的全面配置
        if target_text_modules:
            for i in range(28):
                target_modules.extend([
                    f"model.layers.{i}.self_attn.q_proj",
                    f"model.layers.{i}.self_attn.k_proj",
                    f"model.layers.{i}.self_attn.v_proj",
                    f"model.layers.{i}.self_attn.o_proj",
                    f"model.layers.{i}.mlp.gate_proj",
                    f"model.layers.{i}.mlp.up_proj",
                    f"model.layers.{i}.mlp.down_proj",
                ])
        
        if target_mimo_modules:
            for i in range(8):
                target_modules.extend([
                    f"model.mimo_layers.{i}.self_attn.q_proj",
                    f"model.mimo_layers.{i}.self_attn.k_proj",
                    f"model.mimo_layers.{i}.self_attn.v_proj",
                    f"model.mimo_layers.{i}.self_attn.o_proj",
                    f"model.mimo_layers.{i}.mlp.gate_proj",
                    f"model.mimo_layers.{i}.mlp.up_proj",
                    f"model.mimo_layers.{i}.mlp.down_proj",
                ])
    
    # Audio adapter (always include if target_audio_modules is True)
    if target_audio_modules:
        target_modules.extend([
            "model.vq_adaptor.layers.0",  # First linear layer
            "model.vq_adaptor.layers.3",  # Second linear layer
        ])
    
    # Output heads (always include)
    target_modules.extend([
        "lm_head",
        "mimo_output",
    ])

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha, # Typically set to 2x the rank
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=None, # Don't save full modules, use LoRA for all
    )

    return lora_config

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
        print_lora_info(lora_args, model)

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