from __future__ import annotations
import torch
import torch.nn as nn
from transformers import pipeline, BitsAndBytesConfig, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

class MedGemma(nn.Module):
    """
    MedGemma 4B vision-language model wrapper.
    Frozen at inference — no training.
    """
    model_id = "google/medgemma-4b-it"

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
        ).to(device)
        self.model.eval()
        self.model.generation_config.do_sample = False
        
        # Required for batched generation with padding
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.model.config.pad_token_id = self.processor.tokenizer.eos_token_id
        
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def generate(
        self,
        images: list,           
        messages_batch: list, 
        max_new_tokens: int = 256,
    ) -> list[str]:
        rendered_prompts = []
        prompt_lengths = []

        for messages in messages_batch:
            rendered = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            rendered_prompts.append(rendered)
            tokens = self.processor.tokenizer(
                rendered,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            )
            prompt_lengths.append(tokens["input_ids"].shape[-1])
        
        inputs = self.processor(
            images=[[img] for img in images],
            text=rendered_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_len:]
        responses = self.processor.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )
        return responses