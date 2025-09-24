"""
高级方案：包装KimiAudio模型以利用transformers的GenerationMixin
这个方案创建一个适配器，使得可以使用transformers的generate方法
"""

import torch
import torch.nn as nn
from transformers import GenerationMixin, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, Union


class KimiAudioASRWrapper(nn.Module, GenerationMixin):
    """
    Wrapper class to make KimiAudio compatible with transformers generation methods
    专门用于ASR任务，只处理文本生成
    """
    
    def __init__(self, kimia_model, extra_tokens, kimia_text_audiodelaytokens):
        super().__init__()
        self.kimia_model = kimia_model
        self.extra_tokens = extra_tokens
        self.kimia_text_audiodelaytokens = kimia_text_audiodelaytokens
        
        # Set required attributes for GenerationMixin
        self.config = self._create_generation_config()
        self.device = kimia_model.device
        self.main_input_name = "input_ids"
        
        # Cache for audio inputs (constant during text generation)
        self.cached_audio_ids = None
        self.cached_whisper_feature = None
        self.cached_is_continuous_mask = None
        
    def _create_generation_config(self):
        """Create a GenerationConfig for the wrapper"""
        config = GenerationConfig()
        config.vocab_size = 151936  # Adjust based on actual vocab size
        config.eos_token_id = self.extra_tokens.kimia_text_eos
        config.pad_token_id = self.extra_tokens.kimia_text_blank
        config.bos_token_id = self.extra_tokens.kimia_text_blank
        config.decoder_start_token_id = self.extra_tokens.kimia_text_blank
        return config
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        **kwargs
    ):
        """Prepare inputs for generation step"""
        # For the first step, we need full inputs
        if past_key_values is None:
            return {
                "input_ids": input_ids,
                "past_key_values": past_key_values,
                "use_cache": True,
            }
        else:
            # For subsequent steps, only use the last token
            return {
                "input_ids": input_ids[:, -1:],
                "past_key_values": past_key_values,
                "use_cache": True,
            }
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass that adapts KimiAudio model to standard transformers interface
        This only returns text logits for ASR task
        """
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        device = input_ids.device
        
        # Prepare audio inputs (use blank tokens)
        if past_key_values is None:
            # First step: use cached audio inputs
            audio_input_ids = self.cached_audio_ids
            whisper_feature = self.cached_whisper_feature
            is_continuous_mask = self.cached_is_continuous_mask
            
            # Expand for batch size if needed
            if audio_input_ids.shape[0] != batch_size:
                audio_input_ids = audio_input_ids.expand(batch_size, -1)
                is_continuous_mask = is_continuous_mask.expand(batch_size, -1)
                whisper_feature = [f.expand(batch_size, -1, -1) for f in whisper_feature]
        else:
            # Subsequent steps: use blank audio tokens
            audio_input_ids = torch.full(
                (batch_size, seq_len), 
                self.extra_tokens.kimia_text_blank,
                dtype=torch.long, 
                device=device
            )
            whisper_feature = None
            is_continuous_mask = None
        
        # Create position ids if not provided
        if position_ids is None:
            if past_key_values is None:
                position_ids = torch.arange(
                    audio_input_ids.shape[1], 
                    device=device
                ).unsqueeze(0).expand(batch_size, -1)
            else:
                # Get the position of the last generated token
                past_length = past_key_values[0][0].shape[2]
                position_ids = torch.ones(
                    (batch_size, 1), 
                    device=device, 
                    dtype=torch.long
                ) * past_length
        
        # Call the original KimiAudio model
        audio_logits, text_logits, new_past_key_values = self.kimia_model(
            input_ids=audio_input_ids,
            text_input_ids=input_ids,
            whisper_input_feature=whisper_feature,
            is_continuous_mask=is_continuous_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            return_dict=False,
        )
        
        # Return in transformers format
        if return_dict:
            return CausalLMOutputWithPast(
                logits=text_logits,  # Only return text logits
                past_key_values=new_past_key_values if use_cache else None,
            )
        else:
            return text_logits, new_past_key_values
    
    def set_audio_context(self, audio_ids, whisper_feature, is_continuous_mask):
        """Set the audio context for ASR generation"""
        self.cached_audio_ids = audio_ids
        self.cached_whisper_feature = whisper_feature
        self.cached_is_continuous_mask = is_continuous_mask


def generate_with_transformers(
    kimia_model,
    messages,
    prompt_manager,
    generation_config: Optional[GenerationConfig] = None,
):
    """
    使用transformers的generate方法进行ASR
    
    Args:
        kimia_model: Original KimiAudio model
        messages: Input messages
        prompt_manager: KimiAPromptManager instance
        generation_config: GenerationConfig for controlling generation
    
    Returns:
        Generated text
    """
    # Prepare inputs
    history = prompt_manager.get_prompt(messages, output_type="text")
    audio_input_ids, text_input_ids, is_continuous_mask, _, _ = history.to_tensor()
    audio_features = history.continuous_feature
    
    # Move to device
    device = torch.cuda.current_device()
    audio_input_ids = audio_input_ids.to(device)
    text_input_ids = text_input_ids.to(device)
    is_continuous_mask = is_continuous_mask.to(device)
    audio_features = [f.to(device) for f in audio_features]
    
    # Create wrapper
    wrapper = KimiAudioASRWrapper(
        kimia_model.alm,
        prompt_manager.extra_tokens,
        kimia_model.kimia_text_audiodelaytokens
    )
    
    # Set audio context
    wrapper.set_audio_context(audio_input_ids, audio_features, is_continuous_mask)
    
    # Configure generation
    if generation_config is None:
        generation_config = GenerationConfig(
            max_new_tokens=512,
            num_beams=4,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
            do_sample=False,  # Use beam search
            eos_token_id=prompt_manager.extra_tokens.kimia_text_eos,
            pad_token_id=prompt_manager.extra_tokens.kimia_text_blank,
        )
    
    # Generate using transformers
    with torch.no_grad():
        output_ids = wrapper.generate(
            input_ids=text_input_ids,
            generation_config=generation_config,
        )
    
    # Decode the output
    generated_tokens = output_ids[0, text_input_ids.shape[1]:].cpu().tolist()
    
    # Filter valid tokens
    generated_tokens = [
        t for t in generated_tokens 
        if t < kimia_model.kimia_token_offset and t != prompt_manager.extra_tokens.kimia_text_blank
    ]
    
    # Decode to text
    text = prompt_manager.text_tokenizer.decode(generated_tokens)
    
    return text


# 使用示例
def example_usage():
    from kimia_infer.api.kimia import KimiAudio
    from transformers import GenerationConfig
    
    # 初始化模型
    model = KimiAudio(
        model_path="moonshotai/Kimi-Audio-7B-Instruct",
        load_detokenizer=False,
    )
    
    # 准备消息
    messages = [
        {"role": "user", "message_type": "text", "content": "请将音频内容转换为文字。"},
        {
            "role": "user",
            "message_type": "audio", 
            "content": "test_audios/asr_example.wav",
        },
    ]
    
    # 配置生成参数
    generation_config = GenerationConfig(
        max_new_tokens=512,
        num_beams=5,  # Beam search with 5 beams
        length_penalty=0.8,
        no_repeat_ngram_size=4,
        repetition_penalty=1.2,
        early_stopping=True,
        do_sample=False,  # Deterministic beam search
        temperature=1.0,
        top_k=50,
        top_p=0.95,
    )
    
    # 使用transformers的generate方法
    text = generate_with_transformers(
        model,
        messages,
        model.prompt_manager,
        generation_config=generation_config
    )
    
    print("Generated text:", text)
    
    # 也可以尝试其他生成策略
    # 1. Diverse Beam Search
    generation_config_diverse = GenerationConfig(
        max_new_tokens=512,
        num_beams=6,
        num_beam_groups=3,  # Diverse beam search
        diversity_penalty=0.5,
        length_penalty=1.0,
        early_stopping=True,
    )
    
    # 2. Sampling with top-p (nucleus sampling)
    generation_config_sampling = GenerationConfig(
        max_new_tokens=512,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        top_k=0,  # Disable top-k, use only top-p
        repetition_penalty=1.1,
    )
    
    # 3. Contrastive search
    generation_config_contrastive = GenerationConfig(
        max_new_tokens=512,
        penalty_alpha=0.6,
        top_k=4,
    )


if __name__ == "__main__":
    example_usage()