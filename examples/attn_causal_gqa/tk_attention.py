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

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

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
            
        query_states = query_states.to(torch.bfloat16)
        key_states = key_states.to(torch.bfloat16)
        value_states = value_states.to(torch.bfloat16)
        
        # Regular pytorch attention #
        ##########
        ##########
        key_states_pt = repeat_kv(key_states.permute(0, 2, 1, 3), self.num_heads // self.num_key_value_heads).permute(0, 2, 1, 3)
        value_states_pt = repeat_kv(value_states.permute(0, 2, 1, 3), self.num_heads // self.num_key_value_heads).permute(0, 2, 1, 3)
        
        attn_weights = torch.matmul(query_states, key_states_pt.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        mask = torch.full((q_len, q_len), float('-inf'), device=query_states.device, dtype=query_states.dtype)
        mask = torch.triu(mask, diagonal=1)
        if mask is not None:
            attn_weights = attn_weights + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1).type_as(query_states)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states_pt)
        
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        ##########
        ##########
        
        # Flash attention #
        ##########
        ##########
        # pad seqlen to multiple of 128
        q_len_pad = q_len
        if q_len % 128 != 0:
            q_len_pad = q_len + (128 - q_len % 128)
            query_states = torch.nn.functional.pad(query_states, (0, 0, 0, q_len_pad - q_len), value=0)
            key_states = torch.nn.functional.pad(key_states, (0, 0, 0, q_len_pad - q_len), value=0)
            value_states = torch.nn.functional.pad(value_states, (0, 0, 0, q_len_pad - q_len), value=0)
        
        # attn_output_2 = torch.zeros_like(query_states)
        # mod.attention_forward_causal_gqa(query_states.contiguous(),
        #                                  key_states.contiguous(),
        #                                  value_states.contiguous(),
        #                                  attn_output_2.contiguous())
        
        attn_output_2, _ = scaled_dot_product_gqa(query_states.permute(0, 2, 1, 3).contiguous(), 
                                               key_states.permute(0, 2, 1, 3).contiguous(), 
                                               value_states.permute(0, 2, 1, 3).contiguous(),
                                               is_causal=True,  # default: False
                                               need_weights=False)
        attn_output_2 = attn_output_2.permute(0, 2, 1, 3).contiguous()
        
        # remove padding
        attn_output_2 = attn_output_2[:, :, :q_len, :]
        
        if attn_output_2.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output_2` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output_2.size()}"
            )
        ##########
        ##########
        attn_output_2 = attn_output_2.transpose(1, 2).contiguous()
        attn_output_2 = attn_output_2.reshape(bsz, q_len, -1)
        attn_output_2 = self.o_proj(attn_output_2.float())
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output.float())

        if not output_attentions:
            attn_weights = None
            
        # if avg diff between attn_output and attn_output_2 is > 1e-3, raise error
        avg_diff_mag = torch.mean(torch.abs(attn_output - attn_output_2)).item()

        return attn_output_2, attn_weights, past_key_value
