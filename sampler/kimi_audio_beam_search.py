import os
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple
from loguru import logger
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

from kimia_infer.models.detokenizer import get_audio_detokenizer
from kimia_infer.api.prompt_manager import KimiAPromptManager
from kimia_infer.utils.sampler import KimiASampler


@dataclass
class BeamHypothesis:
    """Beam search hypothesis"""
    token_ids: List[int]
    score: float
    is_finished: bool = False


class KimiAudio(object):
    def __init__(self, model_path: str, load_detokenizer: bool = True):
        logger.info(f"Loading kimi-audio main model")

        if os.path.exists(model_path):
            cache_path = model_path
        else:
            cache_path = snapshot_download(model_path)
    
        logger.info(f"Looking for resources in {cache_path}")
        logger.info(f"Loading whisper model")
        self.alm = AutoModelForCausalLM.from_pretrained(
            cache_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        self.alm = self.alm.to(torch.cuda.current_device())

        model_config = self.alm.config
        self.kimia_text_audiodelaytokens = model_config.kimia_mimo_audiodelaytokens
        self.kimia_token_offset = model_config.kimia_token_offset

        self.prompt_manager = KimiAPromptManager(
            model_path=cache_path, 
            kimia_token_offset=self.kimia_token_offset, 
            kimia_text_audiodelaytokens=self.kimia_text_audiodelaytokens
        )

        if load_detokenizer:
            logger.info(f"Loading detokenizer")
            self.detokenizer = get_audio_detokenizer(cache_path)
        else:
            self.detokenizer = None

        self.extra_tokens = self.prompt_manager.extra_tokens
        self.eod_ids = [self.extra_tokens.msg_end, self.extra_tokens.media_end]

    @torch.inference_mode()
    def _generate_asr_beam_search(
        self,
        audio_input_ids: torch.Tensor,
        text_input_ids: torch.Tensor,
        is_continuous_mask: torch.Tensor,
        continous_feature: torch.Tensor,
        beam_size: int = 4,
        max_new_tokens: int = 512,
        length_penalty: float = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        repetition_window_size: int = 16,
        no_repeat_ngram_size: int = 0,
        early_stopping: bool = True,
    ) -> Tuple[List[int], float]:
        """
        Beam search generation for ASR task (text only)
        
        Args:
            beam_size: Number of beams
            length_penalty: Length penalty factor
            temperature: Temperature for softmax
            repetition_penalty: Repetition penalty factor
            no_repeat_ngram_size: Size of n-grams that cannot be repeated
            early_stopping: Whether to stop when we have beam_size finished hypotheses
        
        Returns:
            Best hypothesis token ids and score
        """
        batch_size = audio_input_ids.shape[0]
        device = audio_input_ids.device
        
        # Initialize beams
        beams = []
        for _ in range(beam_size):
            beams.append(BeamHypothesis(token_ids=[], score=0.0))
        
        # Prepare initial inputs for all beams
        decoder_audio_ids = audio_input_ids.repeat(beam_size, 1)
        decoder_text_ids = text_input_ids.repeat(beam_size, 1)
        decoder_position_ids = torch.arange(
            0, decoder_audio_ids.shape[1], device=device
        ).unsqueeze(0).expand(beam_size, -1).long()
        decoder_whisper_feature = [f.repeat(beam_size, 1, 1) for f in continous_feature]
        decoder_is_continuous_mask = is_continuous_mask.repeat(beam_size, 1)
        
        # Audio tokens for padding (we ignore audio generation for ASR)
        audio_blank_tokens = torch.full(
            (beam_size, 1), self.extra_tokens.kimia_text_blank, 
            dtype=torch.long, device=device
        )
        
        past_key_values = None
        finished_beams = []
        
        for step in range(max_new_tokens):
            # Forward pass
            audio_logits, text_logits, past_key_values = self.alm.forward(
                input_ids=decoder_audio_ids,
                text_input_ids=decoder_text_ids,
                whisper_input_feature=decoder_whisper_feature if step == 0 else None,
                is_continuous_mask=decoder_is_continuous_mask if step == 0 else None,
                position_ids=decoder_position_ids,
                past_key_values=past_key_values,
                return_dict=False,
            )
            
            # Get text logits for the last position
            if len(text_logits.shape) == 3:
                text_logits = text_logits[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty > 1.0:
                for beam_idx, beam in enumerate(beams):
                    if len(beam.token_ids) > 0:
                        recent_tokens = beam.token_ids[-repetition_window_size:]
                        for token in set(recent_tokens):
                            if token < text_logits.shape[-1]:
                                if text_logits[beam_idx, token] < 0:
                                    text_logits[beam_idx, token] *= repetition_penalty
                                else:
                                    text_logits[beam_idx, token] /= repetition_penalty
            
            # Apply no-repeat-ngram constraint
            if no_repeat_ngram_size > 0:
                for beam_idx, beam in enumerate(beams):
                    if len(beam.token_ids) >= no_repeat_ngram_size - 1:
                        ngram_prefix = tuple(beam.token_ids[-(no_repeat_ngram_size-1):])
                        # Check all previous n-grams
                        for i in range(len(beam.token_ids) - no_repeat_ngram_size + 1):
                            prev_ngram = tuple(beam.token_ids[i:i+no_repeat_ngram_size-1])
                            if prev_ngram == ngram_prefix:
                                next_token = beam.token_ids[i+no_repeat_ngram_size-1]
                                if next_token < text_logits.shape[-1]:
                                    text_logits[beam_idx, next_token] = -float('inf')
            
            # Apply temperature and get log probabilities
            if temperature > 0:
                text_logits = text_logits / temperature
            log_probs = F.log_softmax(text_logits, dim=-1)
            
            # Get top-k tokens for each beam
            vocab_size = log_probs.shape[-1]
            next_token_scores = log_probs + torch.tensor(
                [b.score for b in beams], device=device
            ).unsqueeze(1)
            
            # Reshape for getting top-2k candidates
            next_token_scores = next_token_scores.view(1, -1)
            
            # Get top 2*beam_size candidates
            top_scores, top_indices = torch.topk(
                next_token_scores, min(2 * beam_size, next_token_scores.shape[-1]), dim=-1
            )
            
            # Convert to beam and token indices
            beam_indices = top_indices[0] // vocab_size
            token_indices = top_indices[0] % vocab_size
            
            # Create new beams
            new_beams = []
            for rank, (score, beam_idx, token_idx) in enumerate(
                zip(top_scores[0], beam_indices, token_indices)
            ):
                if len(new_beams) >= beam_size:
                    break
                    
                beam_idx = beam_idx.item()
                token_idx = token_idx.item()
                score = score.item()
                
                # Check if this beam should finish
                if token_idx == self.extra_tokens.kimia_text_eos:
                    # Apply length penalty
                    final_score = score / (len(beams[beam_idx].token_ids) + 1) ** length_penalty
                    finished_beams.append(
                        BeamHypothesis(
                            token_ids=beams[beam_idx].token_ids.copy(),
                            score=final_score,
                            is_finished=True
                        )
                    )
                    if early_stopping and len(finished_beams) >= beam_size:
                        break
                else:
                    new_beam = BeamHypothesis(
                        token_ids=beams[beam_idx].token_ids + [token_idx],
                        score=score
                    )
                    new_beams.append(new_beam)
            
            # Check if we should stop
            if early_stopping and len(finished_beams) >= beam_size:
                break
            
            # If all beams are finished
            if len(new_beams) == 0:
                break
            
            # Pad new_beams if necessary
            while len(new_beams) < beam_size:
                new_beams.append(new_beams[-1])  # Duplicate last beam
            
            beams = new_beams
            
            # Prepare inputs for next step
            # Reorder past_key_values according to beam indices
            if past_key_values is not None:
                reordered_past = []
                for layer_past in past_key_values:
                    reordered_layer = []
                    for past_state in layer_past:
                        reordered_layer.append(past_state[beam_indices[:beam_size]])
                    reordered_past.append(tuple(reordered_layer))
                past_key_values = tuple(reordered_past)
            
            # Prepare next step inputs
            decoder_text_ids = torch.tensor(
                [[beams[i].token_ids[-1]] for i in range(beam_size)],
                dtype=torch.long, device=device
            )
            decoder_audio_ids = audio_blank_tokens
            decoder_position_ids = decoder_position_ids[:, -1:] + 1
        
        # Add remaining beams to finished if not already
        for beam in beams:
            if not beam.is_finished:
                final_score = beam.score / len(beam.token_ids) ** length_penalty
                finished_beams.append(
                    BeamHypothesis(
                        token_ids=beam.token_ids,
                        score=final_score,
                        is_finished=True
                    )
                )
        
        # Sort finished beams by score
        finished_beams.sort(key=lambda x: x.score, reverse=True)
        
        # Return best hypothesis
        if finished_beams:
            best_hypothesis = finished_beams[0]
            return best_hypothesis.token_ids, best_hypothesis.score
        else:
            return [], float('-inf')

    @torch.inference_mode()
    def _generate_loop(
        self,
        audio_input_ids: torch.Tensor,
        text_input_ids: torch.Tensor = None,
        max_new_tokens: int = 50,
        audio_top_k: int = 5,
        audio_temperature: float = 0.0,
        audio_repetition_penalty: float = 1.0,
        audio_repetition_window_size: int = 64,
        text_top_k: int = 5,
        text_temperature: float = 0.0,
        text_repetition_penalty: float = 1.0,
        text_repetition_window_size: int = 16,
        is_continuous_mask: torch.Tensor = None,
        continous_feature: torch.Tensor = None,
        output_type: str = "text",
    ):
        """Original generation loop (kept for compatibility)"""
        sampler = KimiASampler(
            audio_top_k=audio_top_k,
            audio_temperature=audio_temperature,
            audio_repetition_penalty=audio_repetition_penalty,
            audio_repetition_window_size=audio_repetition_window_size,
            text_top_k=text_top_k,
            text_temperature=text_temperature,
            text_repetition_penalty=text_repetition_penalty,
            text_repetition_window_size=text_repetition_window_size,
        )

        text_stream_is_finished = False
        previous_audio_tokens = torch.zeros(
            (4096,), dtype=torch.int, device=torch.cuda.current_device()
        )
        text_previous_tokens = torch.zeros(
            (4096,), dtype=torch.int, device=torch.cuda.current_device()
        )

        decoder_input_audio_ids = audio_input_ids.clone()
        decoder_input_text_ids = text_input_ids.clone()
        decoder_position_ids = (
            torch.arange(
                0, decoder_input_audio_ids.shape[1], device=torch.cuda.current_device()
            )
            .unsqueeze(0)
            .long()
        )
        decoder_input_whisper_feature = continous_feature
        decoder_is_continuous_mask = is_continuous_mask
        past_key_values = None

        last_position_id = decoder_input_audio_ids.shape[1] - 1

        valid_text_length = 0
        valid_audio_length = 0

        for i in range(max_new_tokens):
            audio_logits, text_logits, past_key_values = self.alm.forward(
                input_ids=decoder_input_audio_ids,
                text_input_ids=decoder_input_text_ids,
                whisper_input_feature=decoder_input_whisper_feature,
                is_continuous_mask=decoder_is_continuous_mask,
                position_ids=decoder_position_ids,
                past_key_values=past_key_values,
                return_dict=False,
            )

            next_token_text = sampler.sample_text_logits(
                text_logits, recent_tokens=text_previous_tokens[:i] if i > 0 else None
            )

            next_audio_token = sampler.sample_audio_logits(
                audio_logits, recent_tokens=previous_audio_tokens[:i] if i > 0 else None
            )

            if text_stream_is_finished:
                next_token_text.fill_(self.extra_tokens.kimia_text_blank)
            elif next_token_text.item() == self.extra_tokens.kimia_text_eos:
                text_stream_is_finished = True
            else:
                valid_text_length += 1

            text_previous_tokens[i : i + 1] = next_token_text

            if i < self.kimia_text_audiodelaytokens:
                next_audio_token.fill_(self.extra_tokens.kimia_text_blank)
            else:
                if output_type == "text":
                    next_audio_token.fill_(self.extra_tokens.kimia_text_blank)
                else:
                    valid_audio_length += 1

            previous_audio_tokens[i : i + 1] = next_audio_token

            audio_stream_is_finished = next_audio_token.item() in self.eod_ids

            if (
                output_type == "text"
                and text_stream_is_finished
                or output_type == "both"
                and audio_stream_is_finished
            ):
                return_text_tokens = (
                    text_previous_tokens[:valid_text_length]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                return_audio_tokens = (
                    previous_audio_tokens[
                        self.kimia_text_audiodelaytokens : valid_audio_length
                        + self.kimia_text_audiodelaytokens
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                return return_audio_tokens, return_text_tokens
            else:
                decoder_input_audio_ids = next_audio_token.unsqueeze(1)
                decoder_input_text_ids = next_token_text.unsqueeze(1)

                decoder_position_ids = (
                    torch.zeros(1, 1, device=torch.cuda.current_device())
                    .fill_(last_position_id + 1)
                    .long()
                    .view(1, 1)
                )
                last_position_id += 1

                decoder_input_whisper_feature = None
                decoder_is_continuous_mask = None

        return_text_tokens = (
            text_previous_tokens[:valid_text_length].detach().cpu().numpy().tolist()
        )
        return_audio_tokens = (
            previous_audio_tokens[
                self.kimia_text_audiodelaytokens : valid_audio_length
                + self.kimia_text_audiodelaytokens
            ]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )
        print(f"[WARNING] 达到max token限制: {max_new_tokens}")
        return return_audio_tokens, return_text_tokens

    @torch.inference_mode()
    def generate(
        self,
        chats: list[dict],
        output_type="text",
        audio_temperature=0.0,
        audio_top_k=5,
        text_temperature=0.0,
        text_top_k=5,
        audio_repetition_penalty=1.0,
        audio_repetition_window_size=64,
        text_repetition_penalty=1.0,
        text_repetition_window_size=16,
        max_new_tokens=-1,
        # Beam search parameters for ASR
        use_beam_search=False,
        beam_size=4,
        length_penalty=1.0,
        no_repeat_ngram_size=0,
        early_stopping=True,
    ):
        """
        Generate text and/or audio output.
        
        Args:
            use_beam_search: Whether to use beam search for ASR tasks (only for output_type="text")
            beam_size: Number of beams for beam search
            length_penalty: Length penalty for beam search
            no_repeat_ngram_size: Size of n-grams that cannot be repeated
            early_stopping: Whether to stop early in beam search
        """
        assert output_type in ["text", "both"]

        history = self.prompt_manager.get_prompt(chats, output_type=output_type)

        audio_input_ids, text_input_ids, is_continuous_mask, _, _ = history.to_tensor()
        audio_features = history.continuous_feature

        generated_wav_tokens = []
        generated_text_tokens = []

        if output_type == "both":
            max_new_tokens = int(12.5 * 120) - audio_input_ids.shape[1]
            use_beam_search = False  # Beam search only for text output
        else:
            if max_new_tokens == -1:
                max_new_tokens = 512 if use_beam_search else 7500 - audio_input_ids.shape[1]

        audio_input_ids = audio_input_ids.to(torch.cuda.current_device())
        text_input_ids = text_input_ids.to(torch.cuda.current_device())
        is_continuous_mask = is_continuous_mask.to(torch.cuda.current_device())
        audio_features = [f.to(torch.cuda.current_device()) for f in audio_features]

        # Use beam search for ASR if requested
        if use_beam_search and output_type == "text":
            logger.info(f"Using beam search with beam_size={beam_size}")
            generated_text_tokens, score = self._generate_asr_beam_search(
                audio_input_ids=audio_input_ids,
                text_input_ids=text_input_ids,
                is_continuous_mask=is_continuous_mask,
                continous_feature=audio_features,
                beam_size=beam_size,
                max_new_tokens=max_new_tokens,
                length_penalty=length_penalty,
                temperature=text_temperature,
                repetition_penalty=text_repetition_penalty,
                repetition_window_size=text_repetition_window_size,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping,
            )
            logger.info(f"Beam search completed with score: {score:.4f}")
            generated_wav_tokens = []  # No audio for ASR
        else:
            # Use original generation method
            generated_wav_tokens, generated_text_tokens = self._generate_loop(
                audio_input_ids=audio_input_ids,
                text_input_ids=text_input_ids,
                max_new_tokens=max_new_tokens,
                audio_temperature=audio_temperature,
                audio_top_k=audio_top_k,
                audio_repetition_penalty=audio_repetition_penalty,
                audio_repetition_window_size=audio_repetition_window_size,
                text_top_k=text_top_k,
                text_temperature=text_temperature,
                text_repetition_penalty=text_repetition_penalty,
                text_repetition_window_size=text_repetition_window_size,
                is_continuous_mask=is_continuous_mask,
                continous_feature=audio_features,
                output_type=output_type,
            )

        # Process generated tokens
        generated_wav_tokens = [
            t for t in generated_wav_tokens if t >= self.kimia_token_offset
        ]

        generated_wav_tokens = torch.tensor(generated_wav_tokens).unsqueeze(0)
        generated_wav_tokens = generated_wav_tokens - self.kimia_token_offset

        generated_text_tokens = [
            t for t in generated_text_tokens if t < self.kimia_token_offset
        ]
        generated_text = self.detokenize_text(generated_text_tokens)
        
        if self.detokenizer is not None and output_type == "both":
            generated_wav = self.detokenize_audio(generated_wav_tokens)
        else:
            generated_wav = None

        return generated_wav, generated_text

    def detokenize_audio(self, audio_tokens):
        if self.detokenizer is None:
            raise ValueError("Detokenizer is not initialized")
        self.detokenizer.clear_states()
        chunk_size = 30
        first_chunk_size = 30
        cache_speech_collection = []
        audio_tokens = audio_tokens.to(torch.cuda.current_device())
        audio_tokens = audio_tokens.long()
        num_audio_tokens = audio_tokens.size(1)
        first_chunk_semantic_tokens = audio_tokens[:, :first_chunk_size]
        gen_speech = self.detokenizer.detokenize_streaming(
            first_chunk_semantic_tokens,
            is_final=(num_audio_tokens <= first_chunk_size),
            upsample_factor=4,
        )
        cache_speech_collection.append(gen_speech)

        if num_audio_tokens > first_chunk_size:
            res_semantic_tokens = audio_tokens[:, first_chunk_size:]
            for i in range(0, res_semantic_tokens.size(1), chunk_size):
                chunk_semantic_tokens = res_semantic_tokens[:, i : i + chunk_size]
                gen_speech = self.detokenizer.detokenize_streaming(
                    chunk_semantic_tokens,
                    upsample_factor=4,
                    is_final=(i + chunk_size >= res_semantic_tokens.size(1)),
                )
                cache_speech_collection.append(gen_speech)

        gen_speech = torch.cat(cache_speech_collection, dim=-1)
        return gen_speech

    def detokenize_text(self, text_tokens):
        valid_text_ids = []
        for x in text_tokens:
            if x == self.extra_tokens.kimia_text_eos:
                break
            valid_text_ids.append(x)
        return self.prompt_manager.text_tokenizer.decode(valid_text_ids)