import torch
from kimia_infer.api.kimia import KimiAudio
import os
import argparse


def count_parameters(model):
    """统计模型的参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyze_model_structure(model):
    """分析模型结构并统计各个模块的参数量"""
    result = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if params > 0:
            result[name] = params
            # 递归统计子模块
            if len(list(module.children())) > 0:
                sub_modules = {}
                for sub_name, sub_module in module.named_children():
                    sub_params = sum(p.numel() for p in sub_module.parameters() if p.requires_grad)
                    if sub_params > 0:
                        sub_modules[sub_name] = sub_params
                if sub_modules:
                    result[f"{name}_modules"] = sub_modules
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="moonshotai/Kimi-Audio-7B-Instruct")
    args = parser.parse_args()

    # 加载模型
    print(f"正在加载模型 {args.model_path}...")
    model = KimiAudio(
        model_path=args.model_path,
        load_detokenizer=True,
    )

    # 获取总参数量
    total_params = count_parameters(model.alm)
    print(f"\n总参数量: {total_params:,} ({total_params/1e9:.2f}B)")

    # 分析主要模块
    print("\n主要模块参数量:")
    modules = analyze_model_structure(model.alm)
    for name, params in modules.items():
        if not name.endswith("_modules"):
            print(f"- {name}: {params:,} ({params/1e9:.2f}B)")

    # 分析子模块
    print("\n主要子模块详情:")
    for name, params in modules.items():
        if name.endswith("_modules"):
            parent_name = name[:-8]  # 去掉"_modules"后缀
            print(f"\n{parent_name}的子模块:")
            for sub_name, sub_params in params.items():
                print(f"  - {sub_name}: {sub_params:,} ({sub_params/1e9:.2f}B)")

    # 分析Whisper模型 (如果存在)
    if hasattr(model, "detokenizer") and model.detokenizer is not None:
        print("\nWhisper模型信息:")
        if hasattr(model.detokenizer, "num_parameters"):
            whisper_params = model.detokenizer.num_parameters()
            print(f"- Whisper参数量: {whisper_params:,} ({whisper_params/1e9:.2f}B)")

    # 打印模型配置信息
    print("\n模型配置信息:")
    config = model.alm.config
    print(f"- hidden_size: {config.hidden_size}")
    print(f"- intermediate_size: {config.intermediate_size}")
    print(f"- num_hidden_layers: {config.num_hidden_layers}")
    print(f"- num_attention_heads: {config.num_attention_heads}")
    print(f"- vocab_size: {config.vocab_size}")
    
    # 打印KimiAudio特有配置
    print("\nKimiAudio特有配置:")
    kimi_attributes = [attr for attr in dir(config) if attr.startswith('kimia_')]
    for attr in kimi_attributes:
        print(f"- {attr}: {getattr(config, attr)}")

# python 你的脚本名.py --model_path moonshotai/Kimi-Audio-7B-Instruct