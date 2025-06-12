import argparse
import os
import json
import multiprocessing as mp
from functools import partial
from typing import List, Dict, Any
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set multiprocessing start method to 'spawn'
mp.set_start_method('spawn', force=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download
import tqdm

from kimia_infer.api.prompt_manager import KimiAPromptManager

def process_chunk(chunk: List[Dict], gpu_id: int, cache_path: str) -> List[Dict]:
    """Process a chunk of data on a specific GPU."""
    # Set GPU device for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(gpu_id)
    
    # Log GPU usage
    logger.info(f"Process {mp.current_process().name} using GPU {gpu_id}")
    logger.info(f"Available GPU: {torch.cuda.get_device_name(gpu_id)}")
    
    # Load model config and prompt manager
    model_config = AutoConfig.from_pretrained(cache_path, trust_remote_code=True)
    prompt_manager = KimiAPromptManager(
        model_path=cache_path,
        kimia_token_offset=model_config.kimia_token_offset,
        kimia_text_audiodelaytokens=model_config.kimia_mimo_audiodelaytokens
    )
    
    processed_data = []
    for data in chunk:
        for msg in data["conversation"]:
            if msg["message_type"] == "audio":
                audio_path = msg["content"]
                audio_tokens = prompt_manager._tokenize_audio(audio_path)
                msg["audio_tokens"] = audio_tokens
        processed_data.append(data)
    
    return processed_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="moonshotai/Kimi-Audio-7B")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use")
    args = parser.parse_args()

    # Check available GPUs
    num_available_gpus = torch.cuda.device_count()
    if num_available_gpus < args.num_gpus:
        logger.warning(f"Requested {args.num_gpus} GPUs but only {num_available_gpus} are available. Using {num_available_gpus} GPUs.")
        args.num_gpus = num_available_gpus

    if os.path.exists(args.model_name_or_path):
        cache_path = args.model_name_or_path
    else:
        cache_path = snapshot_download(args.model_name_or_path)

    # Read input JSONL file
    data = []
    with open(args.input_file, "r") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))

    # Calculate chunk size for each GPU
    chunk_size = len(data) // args.num_gpus
    if len(data) % args.num_gpus != 0:
        chunk_size += 1

    # Split data into chunks
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    logger.info(f"Split data into {len(chunks)} chunks, each with approximately {chunk_size} items")

    # Create a pool of workers
    with mp.Pool(processes=args.num_gpus) as pool:
        # Create partial function with fixed arguments
        process_func = partial(process_chunk, cache_path=cache_path)
        
        # Map GPU IDs to chunks
        gpu_ids = list(range(args.num_gpus))
        
        # Process chunks in parallel
        results = []
        for result in tqdm.tqdm(
            pool.starmap(process_func, zip(chunks, gpu_ids)),
            total=len(chunks),
            desc="Processing chunks"
        ):
            results.extend(result)

    # Write results to output JSONL file
    with open(args.output_file, "w") as f_out:
        for item in results:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
