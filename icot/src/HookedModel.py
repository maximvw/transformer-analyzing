from typing import Callable, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
from fancy_einsum import einsum
import einops
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention,
    GPT2Block,
    GPT2MLP,
)


class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = None

    def forward(self, x):
        return x


def eager_attention_forward(
    module, query, key, value, attention_mask, head_mask=None, **kwargs
):
    # Making sure some functionalities are never used:
    assert not module.is_cross_attention
    assert not module.scale_attn_by_inverse_layer_idx
    assert head_mask is None
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [],
            value.size(-1) ** 0.5,
            dtype=attn_weights.dtype,
            device=attn_weights.device,
        )

    query_length, key_length = query.size(-2), key.size(-2)

    causal_mask = module.bias[:, :, key_length - query_length : key_length, :key_length]
    mask_value = torch.finfo(attn_weights.dtype).min
    # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
    # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
    mask_value = torch.full(
        [], mask_value, dtype=attn_weights.dtype, device=attn_weights.device
    )
    attn_weights = torch.where(
        causal_mask, attn_weights.to(attn_weights.dtype), mask_value
    )

    if attention_mask is not None:
        # Apply the attention mask
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    
    # [batch, n_heads, seq_len (query), seq_len (key)]
    attn_weights = module.hook_attn_pattern(attn_weights)


    # [batch, n_heads, seq_len (query), seq_len (key)]
    attn_weights = module.hook_attn_pattern(attn_weights)
    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = module.attn_dropout(attn_weights)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)

    return attn_output, attn_weights


def hooked_forward_attention(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    past_key_value=None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = False,
    **kwargs,
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:

    # Making sure some functionalities are never used:
    if self.config._attn_implementation != "eager":
        self.config._attn_implementation = "eager"

    assert encoder_hidden_states is None
    assert encoder_attention_mask is None
    assert output_attentions is False
    assert self.config._attn_implementation == "eager"
    assert self.reorder_and_upcast_attn is False

    query_states, key_states, value_states = self.c_attn(hidden_states).split(
        self.split_size, dim=2
    )

    shape_q = (*query_states.shape[:-1], -1, self.head_dim)
    shape_kv = (*key_states.shape[:-1], -1, self.head_dim)

    query_states = query_states.view(shape_q).transpose(1, 2)
    key_states = key_states.view(shape_kv).transpose(1, 2)

    # [batch, seq_len, n_heads, head_dim]
    value_states = value_states.view(shape_kv)
    value_states = self.hook_value_states(value_states)
    # [batch, n_heads, seq_len, head_dim]
    value_states = value_states.transpose(1, 2)

    if past_key_value is not None:
        cache_kwargs = {"cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs=cache_kwargs
        )

    is_causal = attention_mask is None and query_states.shape[-2] > 1

    attn_output, attn_weights = eager_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        head_mask=head_mask,
        dropout=self.attn_dropout.p if self.training else 0.0,
        is_causal=is_causal,
        **kwargs,
    )
    
    # Original HuggingFace implementation:
    # Useful to uncomment and check against decomposed version of computing attn_output.
    # Something like:
    # assert torch.all(torch.isclose(orig_attn_output, attn_output, atol=1e-6))
    #orig_attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
    #orig_attn_output = self.c_proj(orig_attn_output)
    #orig_attn_output = self.resid_dropout(orig_attn_output)

    attn_output_per_head = einsum(
        "batch seq n_heads d_head, n_heads d_head d_model -> batch seq n_heads d_model",
        attn_output,
        self.W_O,
    )
    attn_output_per_head = self.hook_attn_output_per_head(attn_output_per_head)
    attn_output = attn_output_per_head.sum(dim=2)
    attn_output += self.c_proj.bias
    attn_output = self.resid_dropout(attn_output)
    return attn_output, attn_weights


def hooked_forward_mlp(self, hidden_states):
    """
    Hooked forward pass for GPT2MLP.
    """
    hidden_states = self.c_fc(hidden_states)
    hidden_states = self.hook_mlp_mid(self.act(hidden_states))
    hidden_states = self.c_proj(hidden_states)
    hidden_states = self.dropout(hidden_states)
    return hidden_states


def hooked_forward_block(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    past_key_value=None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
    **kwargs,
) -> Union[
    Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]
]:
    # There are some functionalities that we don't need.
    # Making sure they're never used:
    assert encoder_hidden_states is None
    assert encoder_attention_mask is None
    assert output_attentions is False

    residual = self.hook_resid_pre(hidden_states)
    hidden_states = self.ln_1(hidden_states)
    attn_output, self_attn_weights = self.attn(
        hidden_states,
        past_key_value=past_key_value,
        cache_position=cache_position,
        attention_mask=attention_mask,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        **kwargs,
    )
    # residual connection
    hidden_states = attn_output + residual
    hidden_states = self.hook_resid_mid(hidden_states)

    residual = hidden_states
    hidden_states = self.ln_2(hidden_states)
    feed_forward_hidden_states = self.mlp(hidden_states)
    # residual connection
    hidden_states = residual + feed_forward_hidden_states

    hidden_states = self.hook_resid_post(hidden_states)
    outputs = (hidden_states,)
    return outputs


def _convert_to_hooked_model(module):
    for child in module.children():

        if isinstance(child, GPT2Attention):
            child.forward = hooked_forward_attention.__get__(child, GPT2Attention)

        if isinstance(child, GPT2MLP):
            child.forward = hooked_forward_mlp.__get__(child, GPT2MLP)

        if isinstance(child, GPT2Block):
            child.forward = hooked_forward_block.__get__(child, GPT2Block)
        _convert_to_hooked_model(child)


def convert_to_hooked_model(model):
    """
    This function sets up a hook for the model's forward pass.
    Feel free to add any hooks as you see fit.
    You may need to modify the forward pass of each module above.
    """
    n_heads = model.config.base_model["n_head"]
    d_model = model.config.base_model["n_embd"]
    assert d_model % n_heads == 0
    d_head = d_model // n_heads

    # [V, d_model]
    model.W_E = model.base_model.transformer.wte.weight
    model.W_U = model.base_model.lm_head.weight
    for layer in model.base_model.transformer.h:

        # resid mid.
        layer.hook_resid_pre = HookPoint()
        layer.hook_resid_mid = HookPoint()
        layer.hook_resid_post = HookPoint()

        # Attention.
        layer.attn.hook_attn_pattern = HookPoint()
        layer.attn.hook_attn_output_per_head = HookPoint()

        W = layer.attn.c_attn.weight
        W_Q, W_K, W_V = torch.tensor_split(W, 3, dim=1)
        layer.attn.W_Q = einops.rearrange(W_Q, "d (n h) -> n d h", n=n_heads)
        layer.attn.W_K = einops.rearrange(W_K, "d (n h) -> n d h", n=n_heads)
        layer.attn.W_V = einops.rearrange(W_V, "d (n h) -> n d h", n=n_heads)
        qkv_bias = layer.attn.c_attn.bias
        qkv_bias = einops.rearrange(
            qkv_bias,
            "(qkv index head) -> qkv index head",
            qkv=3,
            index=n_heads,
            head=d_head,
        )
        layer.attn.b_Q = qkv_bias[0]
        layer.attn.b_K = qkv_bias[1]
        layer.attn.b_V = qkv_bias[2]

        layer.attn.hook_value_states = HookPoint()

        layer.attn.W_O = einops.rearrange(
            layer.attn.c_proj.weight, "(n h) d -> n h d", n=n_heads
        )
        layer.attn.b_O = layer.attn.c_proj.bias

        # MLP.
        layer.mlp.W_in = layer.mlp.c_fc.weight
        layer.mlp.b_in = layer.mlp.c_fc.bias

        layer.mlp.W_out = layer.mlp.c_proj.weight
        layer.mlp.b_out = layer.mlp.c_proj.bias

        layer.mlp.hook_mlp_mid = HookPoint()

    _convert_to_hooked_model(model)