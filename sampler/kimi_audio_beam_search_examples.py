from kimia_infer.api.kimia import KimiAudio
import argparse
import time
from loguru import logger

def test_asr_with_beam_search():
    """测试ASR任务的beam search功能"""
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="moonshotai/Kimi-Audio-7B-Instruct")
    parser.add_argument("--use_beam_search", action="store_true", help="使用beam search进行ASR")
    parser.add_argument("--beam_size", type=int, default=4, help="Beam size")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="Length penalty")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3, help="No repeat n-gram size")
    args = parser.parse_args()

    # 初始化模型（ASR任务不需要detokenizer）
    model = KimiAudio(
        model_path=args.model_path,
        load_detokenizer=False,  # ASR任务不需要音频解码器
    )

    # ASR任务示例1：使用原始采样方法
    logger.info("Testing ASR with original sampling method...")
    messages = [
        {"role": "user", "message_type": "text", "content": "请将音频内容转换为文字。"},
        {
            "role": "user",
            "message_type": "audio",
            "content": "test_audios/asr_example.wav",
        },
    ]

    # 原始采样参数
    sampling_params = {
        "text_temperature": 0.0,  # Greedy decoding
        "text_top_k": 1,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
    }

    start_time = time.time()
    wav, text = model.generate(
        messages, 
        **sampling_params, 
        output_type="text",
        use_beam_search=False  # 使用原始方法
    )
    elapsed_time = time.time() - start_time
    logger.info(f"Original method - Time: {elapsed_time:.2f}s")
    print(">>> Original sampling output: ", text)
    print("-" * 50)

    # ASR任务示例2：使用beam search
    if args.use_beam_search:
        logger.info("Testing ASR with beam search...")
        
        # Beam search参数
        beam_params = {
            "use_beam_search": True,
            "beam_size": args.beam_size,
            "length_penalty": args.length_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "early_stopping": True,
            "text_repetition_penalty": 1.1,
            "text_repetition_window_size": 32,
        }
        
        start_time = time.time()
        wav, text = model.generate(
            messages,
            output_type="text",
            **beam_params
        )
        elapsed_time = time.time() - start_time
        logger.info(f"Beam search - Time: {elapsed_time:.2f}s")
        print(">>> Beam search output: ", text)
        print("-" * 50)

    # ASR任务示例3：批量处理多个音频文件
    logger.info("Testing batch ASR processing...")
    audio_files = [
        "test_audios/asr_example.wav",
        "test_audios/qa_example.wav",
        # 添加更多音频文件路径
    ]
    
    for audio_file in audio_files:
        messages = [
            {"role": "user", "message_type": "text", "content": "请准确转写以下音频内容："},
            {
                "role": "user",
                "message_type": "audio",
                "content": audio_file,
            },
        ]
        
        # 使用beam search获得更准确的结果
        wav, text = model.generate(
            messages,
            output_type="text",
            use_beam_search=True,
            beam_size=5,
            length_penalty=0.8,  # 稍微偏向较短的输出
            no_repeat_ngram_size=4,  # 避免重复的4-gram
            text_temperature=0.8,  # 稍微增加随机性
            early_stopping=True,
        )
        
        print(f"File: {audio_file}")
        print(f"Transcription: {text}")
        print("-" * 30)


def compare_decoding_strategies():
    """比较不同解码策略的效果"""
    
    model = KimiAudio(
        model_path="moonshotai/Kimi-Audio-7B-Instruct",
        load_detokenizer=False,
    )
    
    messages = [
        {"role": "user", "message_type": "text", "content": "请将音频内容准确转换为文字。"},
        {
            "role": "user", 
            "message_type": "audio",
            "content": "test_audios/asr_example.wav",
        },
    ]
    
    strategies = [
        {
            "name": "Greedy Decoding",
            "params": {
                "use_beam_search": False,
                "text_temperature": 0.0,
                "text_top_k": 1,
            }
        },
        {
            "name": "Top-K Sampling (k=5)",
            "params": {
                "use_beam_search": False,
                "text_temperature": 0.8,
                "text_top_k": 5,
            }
        },
        {
            "name": "Beam Search (beam=3)",
            "params": {
                "use_beam_search": True,
                "beam_size": 3,
                "length_penalty": 1.0,
            }
        },
        {
            "name": "Beam Search (beam=5) with constraints",
            "params": {
                "use_beam_search": True,
                "beam_size": 5,
                "length_penalty": 0.9,
                "no_repeat_ngram_size": 3,
                "text_repetition_penalty": 1.2,
            }
        },
    ]
    
    print("=" * 60)
    print("Comparing Different Decoding Strategies for ASR")
    print("=" * 60)
    
    for strategy in strategies:
        print(f"\nStrategy: {strategy['name']}")
        print("-" * 40)
        
        start_time = time.time()
        wav, text = model.generate(
            messages,
            output_type="text",
            **strategy['params']
        )
        elapsed_time = time.time() - start_time
        
        print(f"Output: {text}")
        print(f"Time: {elapsed_time:.2f}s")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # 运行主要测试
    test_asr_with_beam_search()
    
    # 运行策略比较（可选）
    # compare_decoding_strategies()