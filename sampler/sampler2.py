"""
简化版：使用transformers的beam search功能进行ASR
修复了whisper特征维度问题
"""

import torch
import torch.nn as nn
from transformers import GenerationConfig
from typing import Optional, Dict, Any, List
import numpy as np
from loguru import logger


class SimpleKimiAudioASR:
    """
    简化的ASR接口，利用beam search进行文本生成
    """
    
    def __init__(self, kimia_model, prompt_manager):
        self.kimia_model = kimia_model
        self.prompt_manager = prompt_manager
        self.device = torch.cuda.current_device()
        
    def generate_with_beam_search(
        self,
        messages: list,
        beam_size: int = 4,
        max_new_tokens: int = 512,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> str:
        """
        使用自定义的beam search进行ASR（借鉴transformers的实现）
        """
        
        # 准备输入
        history = self.prompt_manager.get_prompt(messages, output_type="text")
        audio_input_ids, text_input_ids, is_continuous_mask, _, _ = history.to_tensor()
        audio_features = history.continuous_feature
        
        # 移动到GPU
        audio_input_ids = audio_input_ids.to(self.device)
        text_input_ids = text_input_ids.to(self.device)
        is_continuous_mask = is_continuous_mask.to(self.device)
        
        # 处理audio_features - 确保正确的维度
        if audio_features:
            processed_features = []
            for f in audio_features:
                f = f.to(self.device)
                # 如果特征有batch维度，去除它
                if f.dim() == 3 and f.shape[0] == 1:
                    f = f.squeeze(0)  # [1, seq_len, feature_dim] -> [seq_len, feature_dim]
                processed_features.append(f)
            audio_features = processed_features
        
        # 执行beam search
        with torch.no_grad():
            generated_tokens = self._beam_search(
                audio_input_ids=audio_input_ids,
                text_input_ids=text_input_ids,
                is_continuous_mask=is_continuous_mask,
                audio_features=audio_features,
                beam_size=beam_size,
                max_new_tokens=max_new_tokens,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )
        
        # 解码文本
        return self._decode_tokens(generated_tokens)
    
    def _prepare_whisper_features_for_batch(self, audio_features: List[torch.Tensor], beam_size: int) -> List[torch.Tensor]:
        """
        为beam search准备whisper特征，确保正确的批次处理
        """
        if not audio_features:
            return audio_features
        
        expanded_features = []
        for feat in audio_features:
            # feat可能是 [seq_len, feature_dim] 或 [1, seq_len, feature_dim]
            if feat.dim() == 2:
                # [seq_len, feature_dim] -> [beam_size, seq_len, feature_dim]
                expanded_feat = feat.unsqueeze(0).expand(beam_size, -1, -1)
            elif feat.dim() == 3:
                if feat.shape[0] == 1:
                    # [1, seq_len, feature_dim] -> [beam_size, seq_len, feature_dim]
                    expanded_feat = feat.expand(beam_size, -1, -1)
                else:
                    expanded_feat = feat
            else:
                raise ValueError(f"Unexpected whisper feature dimension: {feat.shape}")
            
            expanded_features.append(expanded_feat)
        
        return expanded_features
    
    def _beam_search(
        self,
        audio_input_ids: torch.Tensor,
        text_input_ids: torch.Tensor,
        is_continuous_mask: torch.Tensor,
        audio_features: list,
        beam_size: int,
        max_new_tokens: int,
        length_penalty: float,
        no_repeat_ngram_size: int,
        temperature: float,
        repetition_penalty: float,
    ) -> list:
        """
        核心beam search实现
        """
        device = audio_input_ids.device
        batch_size = audio_input_ids.shape[0]
        
        # 确保batch_size为1
        if batch_size != 1:
            raise ValueError(f"SimpleKimiAudioASR only supports batch_size=1, got {batch_size}")
        
        # 初始化beam scores
        beam_scores = torch.zeros(beam_size, device=device)
        beam_scores[1:] = -1e9  # 只有第一个beam开始时是活跃的
        
        # 存储每个beam的token序列
        beam_tokens = [[] for _ in range(beam_size)]
        
        # 扩展输入为beam_size
        audio_input_ids = audio_input_ids.repeat(beam_size, 1)
        text_input_ids = text_input_ids.repeat(beam_size, 1) 
        is_continuous_mask = is_continuous_mask.repeat(beam_size, 1)
        
        # 正确处理audio_features的扩展
        audio_features = self._prepare_whisper_features_for_batch(audio_features, beam_size)
        
        # 位置编码
        position_ids = torch.arange(
            audio_input_ids.shape[1], device=device
        ).unsqueeze(0).expand(beam_size, -1)
        
        past_key_values = None
        finished_sequences = []
        
        # 音频空白token
        audio_blank = torch.full(
            (beam_size, 1), 
            self.prompt_manager.extra_tokens.kimia_text_blank,
            dtype=torch.long, device=device
        )
        
        for step in range(max_new_tokens):
            try:
                # 前向传播
                # 第一步传入完整的输入和特征，后续步骤只传入新token
                if step == 0:
                    # 对于第一步，使用完整的输入
                    audio_logits, text_logits, past_key_values = self.kimia_model.alm(
                        input_ids=audio_input_ids,
                        text_input_ids=text_input_ids,
                        whisper_input_feature=audio_features,
                        is_continuous_mask=is_continuous_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        return_dict=False,
                    )
                else:
                    # 后续步骤，只传入最新的token
                    audio_logits, text_logits, past_key_values = self.kimia_model.alm(
                        input_ids=audio_blank,
                        text_input_ids=text_input_ids,
                        whisper_input_feature=None,
                        is_continuous_mask=None,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        return_dict=False,
                    )
            except Exception as e:
                logger.error(f"Error in forward pass at step {step}: {e}")
                logger.error(f"audio_input_ids shape: {audio_input_ids.shape if step == 0 else audio_blank.shape}")
                logger.error(f"text_input_ids shape: {text_input_ids.shape}")
                if audio_features and step == 0:
                    logger.error(f"audio_features[0] shape: {audio_features[0].shape}")
                raise e
            
            # 获取最后一个位置的logits
            if len(text_logits.shape) == 3:
                text_logits = text_logits[:, -1, :]
            
            # 应用温度
            if temperature != 1.0 and temperature > 0:
                text_logits = text_logits / temperature
            
            # 应用重复惩罚
            if repetition_penalty != 1.0:
                for beam_idx in range(beam_size):
                    for token in set(beam_tokens[beam_idx]):
                        if token < text_logits.shape[-1]:
                            if text_logits[beam_idx, token] < 0:
                                text_logits[beam_idx, token] *= repetition_penalty
                            else:
                                text_logits[beam_idx, token] /= repetition_penalty
            
            # 应用n-gram重复限制
            if no_repeat_ngram_size > 0 and step >= no_repeat_ngram_size:
                for beam_idx in range(beam_size):
                    tokens = beam_tokens[beam_idx]
                    if len(tokens) >= no_repeat_ngram_size - 1:
                        # 获取最后的n-1 gram
                        ngram_prefix = tuple(tokens[-(no_repeat_ngram_size-1):])
                        # 查找所有匹配的n-gram并禁止下一个token
                        for i in range(len(tokens) - no_repeat_ngram_size + 1):
                            if tuple(tokens[i:i+no_repeat_ngram_size-1]) == ngram_prefix:
                                banned_token = tokens[i+no_repeat_ngram_size-1]
                                if banned_token < text_logits.shape[-1]:
                                    text_logits[beam_idx, banned_token] = -float('inf')
            
            # 计算log probabilities
            log_probs = torch.nn.functional.log_softmax(text_logits, dim=-1)
            
            # 计算下一步的分数
            next_scores = log_probs + beam_scores.unsqueeze(1)
            
            # 重塑以获取top候选
            vocab_size = log_probs.shape[-1]
            next_scores = next_scores.view(-1)
            
            # 获取top 2*beam_size个候选
            top_k = min(2 * beam_size, next_scores.shape[0])
            next_scores, next_tokens = torch.topk(next_scores, top_k, largest=True, sorted=True)
            
            # 计算beam索引和token索引
            beam_indices = next_tokens // vocab_size
            token_indices = next_tokens % vocab_size
            
            # 选择下一步的beams
            new_beam_scores = []
            new_beam_tokens = []
            new_beam_indices_list = []
            
            for score, beam_idx, token_idx in zip(next_scores, beam_indices, token_indices):
                beam_idx = beam_idx.item()
                token_idx = token_idx.item()
                
                # 检查是否结束
                if token_idx == self.prompt_manager.extra_tokens.kimia_text_eos:
                    # 计算最终分数（应用长度惩罚）
                    final_score = score.item() / ((len(beam_tokens[beam_idx]) + 1) ** length_penalty)
                    finished_sequences.append({
                        'tokens': beam_tokens[beam_idx].copy(),
                        'score': final_score
                    })
                elif len(new_beam_scores) < beam_size:
                    new_beam_scores.append(score)
                    new_beam_tokens.append(beam_tokens[beam_idx] + [token_idx])
                    new_beam_indices_list.append(beam_idx)
                
                if len(new_beam_scores) >= beam_size:
                    break
            
            # 如果所有序列都结束了
            if len(new_beam_scores) == 0:
                break
            
            # 填充如果需要
            while len(new_beam_scores) < beam_size:
                new_beam_scores.append(new_beam_scores[-1])
                new_beam_tokens.append(new_beam_tokens[-1])
                new_beam_indices_list.append(new_beam_indices_list[-1])
            
            # 更新beam状态
            beam_scores = torch.stack(new_beam_scores)
            beam_tokens = new_beam_tokens
            
            # 重新排序past_key_values
            if past_key_values is not None:
                beam_indices_tensor = torch.tensor(new_beam_indices_list, device=device)
                reordered_past = []
                for layer_past in past_key_values:
                    reordered_layer = []
                    for past_state in layer_past:
                        reordered_layer.append(past_state[beam_indices_tensor])
                    reordered_past.append(tuple(reordered_layer))
                past_key_values = tuple(reordered_past)
            
            # 准备下一步输入
            text_input_ids = torch.tensor(
                [[beam_tokens[i][-1]] for i in range(beam_size)],
                dtype=torch.long, device=device
            )
            position_ids = position_ids[:, -1:] + 1
        
        # 将未完成的序列也加入结果
        for i, tokens in enumerate(beam_tokens):
            if tokens:  # 非空序列
                final_score = beam_scores[i].item() / (len(tokens) ** length_penalty)
                finished_sequences.append({
                    'tokens': tokens,
                    'score': final_score
                })
        
        # 按分数排序
        finished_sequences.sort(key=lambda x: x['score'], reverse=True)
        
        # 返回最佳序列的tokens
        if finished_sequences:
            return finished_sequences[0]['tokens']
        return []
    
    def _decode_tokens(self, tokens: list) -> str:
        """解码token序列为文本"""
        # 过滤有效token
        valid_tokens = [
            t for t in tokens 
            if t < self.kimia_model.kimia_token_offset 
            and t != self.prompt_manager.extra_tokens.kimia_text_blank
            and t != self.prompt_manager.extra_tokens.kimia_text_eos
        ]
        
        # 解码
        if valid_tokens:
            return self.prompt_manager.text_tokenizer.decode(valid_tokens)
        return ""


# 使用示例
def test_simple_wrapper():
    from kimia_infer.api.kimia import KimiAudio
    import time
    
    # 初始化模型
    print("Loading model...")
    model = KimiAudio(
        model_path="moonshotai/Kimi-Audio-7B-Instruct",
        load_detokenizer=False,  # ASR不需要音频解码器
    )
    
    # 创建简化的ASR接口
    asr = SimpleKimiAudioASR(model, model.prompt_manager)
    
    # 准备测试消息
    messages = [
        {"role": "user", "message_type": "text", "content": "请将音频内容转换为文字。"},
        {
            "role": "user",
            "message_type": "audio",
            "content": "test_audios/asr_example.wav",
        },
    ]
    
    print("\n" + "="*60)
    print("Testing ASR with Beam Search")
    print("="*60)
    
    # 测试不同的beam配置
    configs = [
        {"beam_size": 1, "name": "Greedy (beam=1)"},
        {"beam_size": 3, "name": "Beam Search (beam=3)"},
        {"beam_size": 5, "length_penalty": 0.8, "name": "Beam=5 with length penalty"},
        {"beam_size": 4, "no_repeat_ngram_size": 3, "repetition_penalty": 1.2, "name": "Beam=4 with constraints"},
    ]
    
    for config in configs:
        name = config.pop("name")
        print(f"\n{name}:")
        print("-" * 40)
        
        try:
            start_time = time.time()
            text = asr.generate_with_beam_search(messages, **config)
            elapsed = time.time() - start_time
            
            print(f"Output: {text}")
            print(f"Time: {elapsed:.2f}s")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)


if __name__ == "__main__":
    test_simple_wrapper()