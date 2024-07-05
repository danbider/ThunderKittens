import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, Cache, repeat_kv

from typing import Optional, Tuple

import math
import os 
import sys

# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(current_dir, 'build/lib.linux-x86_64-cpython-312'))
import h100_fwd as mod

from grouped_query_attention_pytorch.attention import scaled_dot_product_gqa


class TKLlamaAttention(LlamaAttention):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        query_states = query_states.to(torch.bfloat16)
        key_states = key_states.to(torch.bfloat16)
        value_states = value_states.to(torch.bfloat16)
        
        # Regular pytorch attention #
        ##########
        ##########
        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # if attention_mask is not None:  # no matter the length, we just slice it
        #     causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        #     attn_weights = attn_weights + causal_mask

        # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.bfloat16).to(torch.bfloat16)
        # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # attn_output = torch.matmul(attn_weights, value_states)
        
        # if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        #     raise ValueError(
        #         f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
        #         f" {attn_output.size()}"
        #     )
        # attn_output = attn_output.transpose(1, 2).contiguous()
        
        # attn_output = attn_output.reshape(bsz, q_len, -1)
        # attn_output = self.o_proj(attn_output.to(torch.float32))
        ##########
        ##########
        
        # Flash attention #
        ##########
        ##########
        attn_output = torch.nn.functional.scaled_dot_product_attention(query_states, 
                                                                       key_states, 
                                                                       value_states, is_causal=True)
        
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output.to(torch.float32))
        ##########
        ##########

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
