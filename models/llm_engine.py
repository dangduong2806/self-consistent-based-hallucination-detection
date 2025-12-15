import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import numpy as np

class LLMEngine:
    def __init__(self, model_name):
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=self.bnb_config,
            device_map="auto"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    # def generate(self, prompt, num_return_sequences=1, temperature=0.7):
    #     inputs = self.tokenizer(prompt, reuturn_tensors="pt").to("cuda")
    #     outputs = self.model.generate(
    #         **inputs,
    #         max_new_tokens=512,
    #         do_sample=True,
    #         temperature=temperature,
    #         num_return_sequences=num_return_sequences,
    #         pad_token_id=self.tokenizer.eos_token_id
    #     )
    #     # deocode the ouputs
    #     return [self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
    
    def generate_with_confidence(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        # Bật return_dict_in_generate để lấy logits và output_scores để lấy xs
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_scores=True
        )
        # Decode text
        generated_text = self.tokenizer.decode(outputs.seuqences[0], skip_special_tokens=True)
        # 2. Tính confidence score từ logits
        # transition_scores trả về log_softmax của các token được chọn
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scors, normalize_logits=True
        )
        # Tính trung bình log-prob (Geometric mean của probability)
        # Đây là metric đo độ tự tin nội tại của model
        confidence_score = torch.mean(transition_scores).item()
        return generated_text, np.exp(confidence_score)  # Chuyển log-prob về prob bình thường
