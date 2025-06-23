def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, 
    output_dir: str, 
    bias: str = "none",
    save_lora_only: bool = False,
    merge_lora: bool = False,
    lora_args: Optional[LoraArguments] = None
):
    """
    统一的模型保存函数，支持普通模型和LoRA模型的保存
    
    Args:
        trainer: Hugging Face Trainer对象
        output_dir: 输出目录
        bias: LoRA bias配置
        save_lora_only: 是否只保存LoRA权重
        merge_lora: 是否合并LoRA权重到基础模型
        lora_args: LoRA配置参数
    """
    
    # 判断是否使用了LoRA
    is_lora_model = hasattr(trainer.model, 'peft_config') or (
        hasattr(trainer.model, 'base_model') and hasattr(trainer.model.base_model, 'peft_config')
    )
    
    # DeepSpeed Zero3 特殊处理
    if deepspeed.is_deepspeed_zero3_enabled():
        if is_lora_model and not merge_lora:
            # Zero3 + LoRA: 需要特殊处理
            rank0_print("Saving LoRA weights under DeepSpeed Zero3...")
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            # Zero3 + 普通模型或需要合并的LoRA
            state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if is_lora_model:
            if save_lora_only and not merge_lora:
                # 只获取LoRA权重
                state_dict = get_peft_state_maybe_zero_3(
                    trainer.model.named_parameters(), bias
                )
            elif merge_lora:
                # 合并LoRA权重到基础模型
                rank0_print("Merging LoRA weights to base model...")
                merged_model = trainer.model.merge_and_unload()
                state_dict = merged_model.state_dict()
            else:
                # 获取完整的PEFT模型state_dict
                state_dict = trainer.model.state_dict()
        else:
            # 普通模型
            state_dict = trainer.model.state_dict()
    
    # 保存模型
    if trainer.args.should_save and trainer.args.local_rank == 0:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        if is_lora_model:
            if save_lora_only and not merge_lora:
                # 只保存LoRA权重
                rank0_print(f"Saving LoRA weights to {output_dir}")
                _save_lora_weights(trainer.model, state_dict, output_dir, lora_args)
            elif merge_lora:
                # 保存合并后的完整模型
                rank0_print(f"Saving merged model to {output_dir}")
                trainer._save(output_dir, state_dict=state_dict)
                
                # 同时保存tokenizer和配置
                if hasattr(trainer, 'tokenizer') and trainer.tokenizer is not None:
                    trainer.tokenizer.save_pretrained(output_dir)
                
                # 保存模型配置
                if hasattr(trainer.model, 'config'):
                    trainer.model.config.save_pretrained(output_dir)
                elif hasattr(trainer.model, 'base_model') and hasattr(trainer.model.base_model, 'config'):
                    trainer.model.base_model.config.save_pretrained(output_dir)
            else:
                # 保存完整的PEFT模型（包含adapter配置）
                rank0_print(f"Saving PEFT model to {output_dir}")
                trainer.model.save_pretrained(output_dir)
                
                # 保存tokenizer
                if hasattr(trainer, 'tokenizer') and trainer.tokenizer is not None:
                    trainer.tokenizer.save_pretrained(output_dir)
        else:
            # 普通模型保存
            rank0_print(f"Saving model to {output_dir}")
            trainer._save(output_dir, state_dict=state_dict)


def _save_lora_weights(model, state_dict, output_dir, lora_args=None):
    """
    保存LoRA权重和配置
    
    Args:
        model: PEFT模型
        state_dict: LoRA权重字典
        output_dir: 输出目录
        lora_args: LoRA配置参数
    """
    # 保存LoRA权重
    torch.save(state_dict, os.path.join(output_dir, "adapter_model.bin"))
    
    # 保存PEFT配置
    if hasattr(model, 'peft_config'):
        peft_config = model.peft_config
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'peft_config'):
        peft_config = model.base_model.peft_config
    else:
        peft_config = None
    
    if peft_config:
        # 保存adapter配置
        for adapter_name, config in peft_config.items():
            config.save_pretrained(output_dir)
            break  # 通常只有一个adapter
    
    # 如果提供了lora_args，也保存一份用于参考
    if lora_args:
        lora_config_dict = {
            "r": lora_args.lora_r,
            "lora_alpha": lora_args.lora_alpha,
            "lora_dropout": lora_args.lora_dropout,
            "target_modules": lora_args.lora_target_modules,
            "modules_to_save": lora_args.modules_to_save,
            "bias": lora_args.lora_bias,
            "task_type": "CAUSAL_LM",
        }
        with open(os.path.join(output_dir, "lora_config.json"), "w") as f:
            json.dump(lora_config_dict, f, indent=2)
    
    rank0_print(f"LoRA weights saved to {output_dir}")


# 修改训练函数中的保存逻辑
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

    # ... [保持原有的初始化代码不变] ...

    trainer.train()
    trainer.save_state()

    # 统一的模型保存逻辑
    if training_args.use_lora:
        # LoRA模型保存策略
        if lora_args.save_strategy == "lora_only":
            # 只保存LoRA权重
            safe_save_model_for_hf_trainer(
                trainer=trainer,
                output_dir=os.path.join(training_args.output_dir, "lora_weights"),
                bias=lora_args.lora_bias,
                save_lora_only=True,
                merge_lora=False,
                lora_args=lora_args
            )
        elif lora_args.save_strategy == "merged":
            # 保存合并后的模型
            safe_save_model_for_hf_trainer(
                trainer=trainer,
                output_dir=os.path.join(training_args.output_dir, "merged_model"),
                bias=lora_args.lora_bias,
                save_lora_only=False,
                merge_lora=True,
                lora_args=lora_args
            )
        elif lora_args.save_strategy == "both":
            # 同时保存LoRA权重和合并后的模型
            # 1. 保存LoRA权重
            safe_save_model_for_hf_trainer(
                trainer=trainer,
                output_dir=os.path.join(training_args.output_dir, "lora_weights"),
                bias=lora_args.lora_bias,
                save_lora_only=True,
                merge_lora=False,
                lora_args=lora_args
            )
            # 2. 保存合并后的模型
            safe_save_model_for_hf_trainer(
                trainer=trainer,
                output_dir=os.path.join(training_args.output_dir, "merged_model"),
                bias=lora_args.lora_bias,
                save_lora_only=False,
                merge_lora=True,
                lora_args=lora_args
            )
        else:
            # 默认：保存完整的PEFT模型
            safe_save_model_for_hf_trainer(
                trainer=trainer,
                output_dir=training_args.output_dir,
                bias=lora_args.lora_bias,
                save_lora_only=False,
                merge_lora=False,
                lora_args=lora_args
            )
    else:
        # 普通模型保存
        safe_save_model_for_hf_trainer(
            trainer=trainer,
            output_dir=training_args.output_dir,
            bias="none"
        )


# 扩展LoraArguments以支持保存策略配置
@dataclass
class LoraArguments:
    # ... [保持原有的LoRA参数] ...
    
    save_strategy: str = field(
        default="default",
        metadata={
            "help": "LoRA model save strategy: 'default' (save PEFT model), "
                   "'lora_only' (save only LoRA weights), "
                   "'merged' (save merged model), "
                   "'both' (save both LoRA weights and merged model)"
        }
    )
    merge_lora_after_training: bool = field(
        default=False,
        metadata={"help": "Whether to merge LoRA weights after training (deprecated, use save_strategy instead)"}
    )