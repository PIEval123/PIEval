import numpy as np
from .utils import process_attn, calc_attn_score, process_attn_batch, calc_attn_score_batch
import torch

class AttentionDetector():
    def __init__(self, model, pos_examples=None, neg_examples=None, use_token="first", instruction="Say xxxxxx", threshold=0.5):
        self.name = "attention"
        self.attn_func = "normalize_sum"
        self.model = model
        self.important_heads = model.important_heads
        self.instruction = instruction
        self.use_token = use_token
        self.threshold = threshold

    def attn2score(self, attention_maps, input_range):
        if self.use_token == "first":
            attention_maps = [attention_maps[0]]
        scores = []
        for attention_map in attention_maps:
            heatmap = process_attn(
                attention_map, input_range, self.attn_func)
            score = calc_attn_score(heatmap, self.important_heads)
            scores.append(score)

        return sum(scores) if len(scores) > 0 else 0

    def detect(self, data_prompt):
        _, _, attention_maps, _, input_range, _ = self.model.inference(
            self.instruction, data_prompt, max_output_tokens=1)

        focus_score = self.attn2score(attention_maps, input_range)
        return bool(focus_score <= self.threshold), {"focus_score": focus_score}
    
    def detect_with_instruction(self, instruction, data_prompt):
        _, _, attention_maps, _, input_range, _ = self.model.inference(
            instruction, data_prompt, max_output_tokens=1)
        
        focus_score = self.attn2score(attention_maps, input_range)
        return bool(focus_score <= self.threshold), {"focus_score": focus_score}

    def detect_gcg(self, instruction_len, data_len, input_embeds, prefix_cache_batch=None):
        attention_maps, input_range = self.model.inference_gcg(
            instruction_len, data_len, input_embeds, prefix_cache_batch)

        focus_score = self.attn2score_batch(attention_maps, input_range)
        return focus_score


    def attn2score_batch(self, attention_maps, input_range):
        heatmap = process_attn_batch(attention_maps, input_range, self.attn_func)
        score = calc_attn_score_batch(heatmap, self.important_heads)
        return score

    