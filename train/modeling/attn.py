import torch
import torch.nn as nn

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, TransformersKwargs
from typing import Unpack

from modeling.kernel import attn

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    '''
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    '''
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **_: Unpack[TransformersKwargs],
):
    '''
    Flash-Attention Interface.
    dropout must be 0.0, as flash-attention does not support dropout.
    '''

    assert dropout == 0.0, 'Dropout is not supported in flash-attention'
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    return attn(
        query=query,
        key=key_states,
        value=value_states,
        attention_mask=attention_mask,
        scaling=scaling,
    )


ALL_ATTENTION_FUNCTIONS['eternity-attention'] = attention
