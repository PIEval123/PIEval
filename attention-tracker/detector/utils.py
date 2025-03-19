import torch 
import numpy as np

def process_attn(attention, rng, attn_func):
    heatmap = np.zeros((len(attention), attention[0].shape[1]))
    for i, attn_layer in enumerate(attention):
        attn_layer = attn_layer.to(torch.float32).numpy()

        if "sum" in attn_func:
            last_token_attn_to_inst = np.sum(attn_layer[0, :, -1, rng[0][0]:rng[0][1]], axis=1)
            attn = last_token_attn_to_inst
        
        elif "max" in attn_func:
            last_token_attn_to_inst = np.max(attn_layer[0, :, -1, rng[0][0]:rng[0][1]], axis=1)
            attn = last_token_attn_to_inst

        else: raise NotImplementedError
            
        last_token_attn_to_inst_sum = np.sum(attn_layer[0, :, -1, rng[0][0]:rng[0][1]], axis=1)
        last_token_attn_to_data_sum = np.sum(attn_layer[0, :, -1, rng[1][0]:rng[1][1]], axis=1)

        if "normalize" in attn_func:
            epsilon = 1e-8
            heatmap[i, :] = attn / (last_token_attn_to_inst_sum + last_token_attn_to_data_sum + epsilon)
        else:
            heatmap[i, :] = attn

    heatmap = np.nan_to_num(heatmap, nan=0.0)

    return heatmap


def calc_attn_score(heatmap, heads):
    score = np.mean([heatmap[l, h] for l, h in heads], axis=0)
    return score

def process_attn_batch(attention, rng, attn_func):
    batch_size = attention[0].shape[0]
    num_layers = len(attention)
    num_heads = attention[0].shape[1]
    
    # heatmap: [batch_size, num_layers, num_heads]
    heatmap = torch.zeros((batch_size, num_layers, num_heads), dtype=torch.float32, device=attention[0].device)

    for i, attn_layer in enumerate(attention):
        # attn_layer: [batch_size, num_heads, 1, seq_length]
        if "sum" in attn_func:
            last_token_attn_to_inst = attn_layer[:, :, -1, rng[0][0]:rng[0][1]].sum(dim=-1)  # [batch_size, num_heads]
            attn = last_token_attn_to_inst
        
        elif "max" in attn_func:
            last_token_attn_to_inst = attn_layer[:, :, -1, rng[0][0]:rng[0][1]].max(dim=-1).values  # [batch_size, num_heads]
            attn = last_token_attn_to_inst

        else:
            raise NotImplementedError(f"Unsupported attention function: {attn_func}")
        
        # Normalization if required
        if "normalize" in attn_func:
            last_token_attn_to_inst_sum = attn_layer[:, :, -1, rng[0][0]:rng[0][1]].sum(dim=-1)  # [batch_size, num_heads]
            last_token_attn_to_data_sum = attn_layer[:, :, -1, rng[1][0]:rng[1][1]].sum(dim=-1)  # [batch_size, num_heads]
            epsilon = 1e-8
            heatmap[:, i, :] = attn / (last_token_attn_to_inst_sum + last_token_attn_to_data_sum + epsilon)
        else:
            heatmap[:, i, :] = attn

    return heatmap


def calc_attn_score_batch(heatmap, heads):
    selected_scores = torch.stack([heatmap[:, l, h] for l, h in heads], dim=-1)  # [batch_size, len(heads)]
    
    score = selected_scores.mean(dim=-1)
    
    return score
