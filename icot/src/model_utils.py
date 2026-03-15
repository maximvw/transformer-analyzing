import json
import torch
import torch.nn as nn
import einops
from transformers import GPT2Tokenizer
from src.ImplicitModel import ImplicitModel, ImplicitModelConfig
from src.transformer import Transformer, TransformerConfig
import types


def _process_state_dict(state_dict_path):
    state_dict = torch.load(state_dict_path)
    new_state_dict = {}

    for key in list(state_dict.keys()):
        if key.startswith("base_model."):
            new_key = key.replace("base_model.", "")
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]

    return new_state_dict


def _convert(orig_state_dict, cfg):
    state_dict = {}

    state_dict["embed.W_E"] = orig_state_dict["transformer.wte.weight"]
    state_dict["pos_embed.W_pos"] = orig_state_dict["transformer.wpe.weight"]

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = orig_state_dict[
            f"transformer.h.{l}.ln_1.weight"
        ]
        state_dict[f"blocks.{l}.ln1.b"] = orig_state_dict[
            f"transformer.h.{l}.ln_1.bias"
        ]

        # In GPT-2, q,k,v are produced by one big linear map, whose output is
        # concat([q, k, v])
        W = orig_state_dict[f"transformer.h.{l}.attn.c_attn.weight"]
        W_Q, W_K, W_V = torch.tensor_split(W, 3, dim=1)
        W_Q = einops.rearrange(W_Q, "m (i h)->i m h", i=cfg.n_heads)
        W_K = einops.rearrange(W_K, "m (i h)->i m h", i=cfg.n_heads)
        W_V = einops.rearrange(W_V, "m (i h)->i m h", i=cfg.n_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        qkv_bias = orig_state_dict[f"transformer.h.{l}.attn.c_attn.bias"]
        qkv_bias = einops.rearrange(
            qkv_bias,
            "(qkv index head)->qkv index head",
            qkv=3,
            index=cfg.n_heads,
            head=cfg.d_head,
        )
        state_dict[f"blocks.{l}.attn.b_Q"] = qkv_bias[0]
        state_dict[f"blocks.{l}.attn.b_K"] = qkv_bias[1]
        state_dict[f"blocks.{l}.attn.b_V"] = qkv_bias[2]

        W_O = orig_state_dict[f"transformer.h.{l}.attn.c_proj.weight"]
        W_O = einops.rearrange(W_O, "(i h) m->i h m", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = orig_state_dict[
            f"transformer.h.{l}.attn.c_proj.bias"
        ]

        state_dict[f"blocks.{l}.ln2.w"] = orig_state_dict[
            f"transformer.h.{l}.ln_2.weight"
        ]
        state_dict[f"blocks.{l}.ln2.b"] = orig_state_dict[
            f"transformer.h.{l}.ln_2.bias"
        ]

        W_in = orig_state_dict[f"transformer.h.{l}.mlp.c_fc.weight"]
        state_dict[f"blocks.{l}.mlp.W_in"] = W_in
        state_dict[f"blocks.{l}.mlp.b_in"] = orig_state_dict[
            f"transformer.h.{l}.mlp.c_fc.bias"
        ]

        W_out = orig_state_dict[f"transformer.h.{l}.mlp.c_proj.weight"]
        state_dict[f"blocks.{l}.mlp.W_out"] = W_out
        state_dict[f"blocks.{l}.mlp.b_out"] = orig_state_dict[
            f"transformer.h.{l}.mlp.c_proj.bias"
        ]
    state_dict["unembed.W_U"] = orig_state_dict[f"lm_head.weight"].T

    state_dict["ln_final.w"] = orig_state_dict[f"transformer.ln_f.weight"]
    state_dict["ln_final.b"] = orig_state_dict[f"transformer.ln_f.bias"]
    return state_dict


def load_hf_model(config_path, state_dict_path, cpu=False):

    # Load config and state dict
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    config = ImplicitModelConfig(**config_dict)  # Convert dict to config object
    model = ImplicitModel(config)
    if cpu:
        model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
    else:
        model.load_state_dict(torch.load(state_dict_path))
    model.eval()
    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_c_hat_model(model_path):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    device = "cuda"
    config = TransformerConfig(
        hidden_dim=768,
        depth=2,
        n_heads=4,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=128,
        device="cuda",
    )
    model = Transformer(config).to(device)

    # Load the state dict to inspect what aux heads were used
    state_dict = torch.load(model_path, map_location="cuda")

    # Extract aux_heads from the state dict keys
    aux_heads = []
    for key in state_dict.keys():
        if key.startswith("linear_regression_heads.") and key.endswith(".weight"):
            head_id = int(key.split(".")[1])
            aux_heads.append(head_id)

    print(f"Found auxiliary heads: {aux_heads}")

    # Add the linear regression heads
    attn_dim = config.hidden_dim // config.n_heads  # 768 // 4 = 192
    model.linear_regression_heads = torch.nn.ModuleDict(
        {str(h): torch.nn.Linear(attn_dim, 1, bias=False).to(device) for h in aux_heads}
    )

    model.load_state_dict(state_dict)
    model.eval()
    return HFCompat(model), tokenizer


class HFCompat(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x, return_attn=False):
        out = self.base(x, return_attn=return_attn)
        if return_attn:
            logits, attn = out
            return types.SimpleNamespace(logits=logits, attn=attn)
        return types.SimpleNamespace(logits=out)
