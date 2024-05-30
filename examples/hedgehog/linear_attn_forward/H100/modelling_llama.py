import torch
import sys
import os
from torch.cuda import Event
from torch.autograd import Function
import numpy as np

import math
from typing import List, Optional, Tuple, Union

import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.insert(0, project_root)
sys.path.append('build/lib.linux-x86_64-3.10')

import lin_attn_h100 as tk_kernel

from collections import defaultdict
import matplotlib.pyplot as plt
from statistics import median
import torch
import torch.nn as nn

torch.random.manual_seed(42)

from transformers.models.llama.modeling_llama import LlamaAttention

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaHHLinearAttentionTK(LlamaAttention):
    
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self._init_rope()
        
    def feature_qk(self, query_states, key_states):
        q_max = torch.amax(query_states, dim=-1, keepdim=True)
        q = torch.cat([
            torch.exp(query_states - q_max), torch.exp(-query_states + q_max)
        ], dim=-1)
        
        k_max = torch.amax(key_states, dim=-1, keepdim=True)
        k = torch.cat([
            torch.exp(key_states - k_max), torch.exp(-key_states + k_max)
        ], dim=-1)
        
        return q, k
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        timings: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        
        assert(output_attentions is False)
        
        start_event = Event(enable_timing=True)
        end_event = Event(enable_timing=True)
        
        # Project
        start_event.record()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        end_event.record()
        torch.cuda.synchronize()
        proj_time = start_event.elapsed_time(end_event)
        
        # Reshape and transpose
        start_event.record()
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        end_event.record()
        torch.cuda.synchronize()
        reshape_time = start_event.elapsed_time(end_event)

        # Rotary embeddings
        start_event.record()
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        end_event.record()
        torch.cuda.synchronize()
        rotary_time = start_event.elapsed_time(end_event)

        # Update past key value
        if past_key_value is not None:
            start_event.record()
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            end_event.record()
            torch.cuda.synchronize()
            update_time = start_event.elapsed_time(end_event)
        else:
            update_time = 0

        # Repeat and reshape key and value states
        start_event.record()
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        end_event.record()
        torch.cuda.synchronize()
        repeat_time = start_event.elapsed_time(end_event)
        
        # Feature map
        start_event.record()
        q, k = self.feature_qk(query_states, key_states)
        end_event.record()
        torch.cuda.synchronize()
        feature_time = start_event.elapsed_time(end_event)

        # Attention computation
        attn_output = torch.zeros_like(value_states).contiguous()
        kv_state = torch.zeros((bsz, self.num_heads, self.head_dim*2, self.head_dim), dtype=attn_output.dtype, device=attn_output.device).contiguous()
        start_event.record()
        tk_kernel.hh_lin_tk(q.contiguous(), k.contiguous(), value_states.contiguous(), attn_output, kv_state)
        end_event.record()
        torch.cuda.synchronize()
        attn_time = start_event.elapsed_time(end_event)
        
        # Normalize
        start_event.record()
        attn_output = attn_output / (torch.einsum("bhld,bhld->bhl", q, k.float().cumsum(dim=2).bfloat16()))[..., None]
        end_event.record()
        torch.cuda.synchronize()
        norm_time = start_event.elapsed_time(end_event)

        # Reshape output
        start_event.record()
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        end_event.record()
        torch.cuda.synchronize()
        reshape_out_time = start_event.elapsed_time(end_event)

        # Final projection
        start_event.record()
        attn_output = self.o_proj(attn_output)
        end_event.record()
        torch.cuda.synchronize()
        proj_out_time = start_event.elapsed_time(end_event)

        if not output_attentions:
            attn_weights = None
            
        # Record timings
        if timings is not None:
            timings['proj_time'].append(proj_time)
            timings['reshape_time'].append(reshape_time)
            timings['rotary_time'].append(rotary_time)
            timings['update_time'].append(update_time)
            timings['repeat_time'].append(repeat_time)
            timings['feature_time'].append(feature_time)
            timings['attn_time'].append(attn_time)
            timings['norm_time'].append(norm_time)
            timings['reshape_out_time'].append(reshape_out_time)
            timings['proj_out_time'].append(proj_out_time)

        return attn_output, None, past_key_value
    
    def forward_quadratic(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        timings: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        
        assert(output_attentions is False)
        
        # Project
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape and transpose
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Rotary embeddings
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Update past key value
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        else:
            update_time = 0

        # Repeat and reshape key and value states
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Feature map
        q, k = self.feature_qk(query_states, key_states)

        # Attention computation
        a = torch.einsum('bhmd,bhnd->bhmn', q, k)
        if self.is_causal:  # Apply causal mask
            m, n = a.shape[-2:]
            causal_mask = torch.ones((m, n), device=a.device, dtype=torch.bool).triu(n - m + 1)
            a = a.masked_fill(causal_mask, 0)
        attn_output = torch.einsum('bhmn,bhnd->bhmd', a, value_states)

        # Normalize
        attn_output = attn_output / (torch.einsum("bhld,bhld->bhl", q, k.float().cumsum(dim=2).bfloat16()))[..., None]

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # Final projection
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, None, past_key_value
    
    def forward_quadratic_fa2(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        timings: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        
        assert(output_attentions is False)
        
        start_event = Event(enable_timing=True)
        end_event = Event(enable_timing=True)
        
        # Project
        start_event.record()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        end_event.record()
        torch.cuda.synchronize()
        proj_time = start_event.elapsed_time(end_event)
        
        # Reshape and transpose
        start_event.record()
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        end_event.record()
        torch.cuda.synchronize()
        reshape_time = start_event.elapsed_time(end_event)

        # Rotary embeddings
        start_event.record()
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        end_event.record()
        torch.cuda.synchronize()
        rotary_time = start_event.elapsed_time(end_event)

        # Update past key value
        if past_key_value is not None:
            start_event.record()
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            end_event.record()
            torch.cuda.synchronize()
            update_time = start_event.elapsed_time(end_event)
        else:
            update_time = 0

        # Repeat and reshape key and value states
        start_event.record()
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        end_event.record()
        torch.cuda.synchronize()
        repeat_time = start_event.elapsed_time(end_event)
        
        start_event.record()
        attn_output = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, is_causal=self.is_causal)
        end_event.record()
        torch.cuda.synchronize()
        attn_time = start_event.elapsed_time(end_event)
        
        feature_time = 0
        norm_time = 0

        # Reshape output
        start_event.record()
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        end_event.record()
        torch.cuda.synchronize()
        reshape_out_time = start_event.elapsed_time(end_event)

        # Final projection
        start_event.record()
        attn_output = self.o_proj(attn_output)
        end_event.record()
        torch.cuda.synchronize()
        proj_out_time = start_event.elapsed_time(end_event)

        if not output_attentions:
            attn_weights = None
            
        # Record timings
        if timings is not None:
            timings['proj_time'].append(proj_time)
            timings['reshape_time'].append(reshape_time)
            timings['rotary_time'].append(rotary_time)
            timings['update_time'].append(update_time)
            timings['repeat_time'].append(repeat_time)
            timings['feature_time'].append(feature_time)
            timings['attn_time'].append(attn_time)
            timings['norm_time'].append(norm_time)
            timings['reshape_out_time'].append(reshape_out_time)
            timings['proj_out_time'].append(proj_out_time)

        return attn_output, None, past_key_value
    

config = LlamaConfig(
    vocab_size=32000,
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=None,
    hidden_act="silu",
    max_position_embeddings=2048,
    initializer_range=0.02,
    rms_norm_eps=1e-6,
    use_cache=True,
    pad_token_id=None,
    bos_token_id=1,
    eos_token_id=2,
    pretraining_tp=1,
    tie_word_embeddings=False,
    rope_theta=10000.0,
    rope_scaling=None,
    attention_bias=False,
    attention_dropout=0.0,
    mlp_bias=False,
)

llamaAttention = LlamaHHLinearAttentionTK(config).to(device='cuda:0')

llamaAttention.to(dtype=torch.bfloat16)

### Correctness check 
hidden_states = torch.randn((1, 4096, 4096), dtype=torch.bfloat16, device='cuda:0')
position_ids = torch.arange(
    0,
    4096 + 0,
    dtype=torch.long, device=hidden_states.device
).unsqueeze(0)

quad_out, _, _ = llamaAttention.forward_quadratic(hidden_states=hidden_states, position_ids=position_ids)
tk_out, _, _ = llamaAttention(hidden_states=hidden_states, position_ids=position_ids)

# print max diff
print(f"Max diff: {torch.max(torch.abs(quad_out - tk_out))}")

# print out values
# print(f"Quad Out: {quad_out[0, 50:70, :4]}")
# print(f"TK Out: {tk_out[0, 50:70, :4]}")
# print("-" * 80)

def profile_attention(B, N):
    hidden_states = torch.randn((B, N, 4096), dtype=torch.bfloat16, device='cuda:0')
    position_ids = torch.arange(
        0, 
        N + 0, 
        dtype=torch.long, device=hidden_states.device
    ).unsqueeze(0)
    
    # Warm-up
    for _ in range(10):
        llamaAttention.forward_quadratic_fa2(hidden_states=hidden_states, position_ids=position_ids)
        llamaAttention(hidden_states=hidden_states, position_ids=position_ids)

    # Timing
    timings_forward = defaultdict(list)
    timings_forward_quadratic = defaultdict(list)

    start_event = Event(enable_timing=True)
    end_event = Event(enable_timing=True)

    # Profile forward_quadratic_fa2
    start_event.record()
    for _ in range(100):
        quad_out, _, _ = llamaAttention.forward_quadratic_fa2(hidden_states=hidden_states, position_ids=position_ids, timings=timings_forward_quadratic)
    end_event.record()
    torch.cuda.synchronize()
    quad_time = sum([sum(values) for values in timings_forward_quadratic.values()]) / 100

    # Profile the custom kernel
    start_event.record()
    for _ in range(100):
        tk_out, _, _ = llamaAttention(hidden_states=hidden_states, position_ids=position_ids, timings=timings_forward)
    end_event.record()
    torch.cuda.synchronize()
    tk_time = sum([sum(values) for values in timings_forward.values()]) / 100

    return quad_time, tk_time

# Sweep 1: Constant batch size, varying sequence length
B = 1
sequence_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
quad_times = []
tk_times = []

for N in sequence_lengths:
    quad_time, tk_time = profile_attention(B, N)
    quad_times.append(quad_time)
    tk_times.append(tk_time)

plt.figure(figsize=(10, 6))
plt.plot(sequence_lengths, quad_times, label='Quadratic Attention', marker='o')
plt.plot(sequence_lengths, tk_times, label='Custom Kernel', marker='x')
plt.xlabel('Sequence Length')
plt.ylabel('Total Time (ms)')
plt.title('Total Time vs Sequence Length (Constant Batch Size)')
plt.legend()
plt.grid(True)
# save as PNG
plt.savefig('total_time_vs_sequence_length.png')

# look at the breakdown
hidden_states = torch.randn((1, 4096, 4096), dtype=torch.bfloat16, device='cuda:0')
position_ids = torch.arange(
    0, 
    4096 + 0, 
    dtype=torch.long, device=hidden_states.device
).unsqueeze(0)

# Warm-up
for _ in range(10):
    llamaAttention.forward_quadratic_fa2(hidden_states=hidden_states, position_ids=position_ids)
    llamaAttention(hidden_states=hidden_states, position_ids=position_ids)

# Timing
timings_forward = defaultdict(list)
timings_forward_quadratic = defaultdict(list)

start_event = Event(enable_timing=True)
end_event = Event(enable_timing=True)

# Profile forward_quadratic_fa2
start_event.record()
for _ in range(100):
    quad_out, _, _ = llamaAttention.forward_quadratic_fa2(hidden_states=hidden_states, position_ids=position_ids, timings=timings_forward_quadratic)
end_event.record()
torch.cuda.synchronize()
quad_time = sum([sum(values) for values in timings_forward_quadratic.values()]) / 100

# print timings_forward_quadratic breakdown
print("-" * 80)
print(f"Forward Quadratic: {quad_time} ms")
print("-" * 80)
for key, values in timings_forward_quadratic.items():
    print(f"{key}: {sum(values) / 100} ms")
print("-" * 80)


# Profile the custom kernel
start_event.record()
for _ in range(100):
    tk_out, _, _ = llamaAttention(hidden_states=hidden_states, position_ids=position_ids, timings=timings_forward)
end_event.record()
torch.cuda.synchronize()
tk_time = sum([sum(values) for values in timings_forward.values()]) / 100

# print timings_forward breakdown
print(f"Custom Kernel: {tk_time} ms")
print("-" * 80)
for key, values in timings_forward.items():
    print(f"{key}: {sum(values) / 100} ms")
print("-" * 80)