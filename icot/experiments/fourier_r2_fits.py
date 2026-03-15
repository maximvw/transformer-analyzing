from typing import List, Tuple, Mapping
import os
import numpy as np
import torch
from fancy_einsum import einsum

from src.model_utils import load_hf_model
from src.data_utils import prompt_ci_raw_format_batch
from src.HookedModel import convert_to_hooked_model
from src.ActivationCache import record_activations
from constants import BASE_DIR

# %%

model_name = "2L4H"
model_path = os.path.join(BASE_DIR, "ckpts/2L4H")

config_path = os.path.join(model_path, "config.json")
state_dict_path = os.path.join(model_path, "state_dict.bin")

# %%

hooked_model, tokenizer = load_hf_model(config_path, state_dict_path)
hooked_model.to("cuda")
convert_to_hooked_model(hooked_model)


# %%

digits = [f" {i}" for i in range(10)]
tokens = tokenizer(
    digits, return_tensors="pt", padding=False, truncation=True
).input_ids.squeeze()

# %%

embeds = hooked_model.base_model.transformer.wte.weight.detach()
# [10, d_model]
embeds = embeds[tokens]
unembeds = hooked_model.base_model.lm_head.weight
# [10, d_model]
unembeds = unembeds[tokens]

# %%


# [d_mlp, d_model]
W_out = hooked_model.base_model.transformer.h[1].mlp.c_proj.weight.detach()
W_in = hooked_model.base_model.transformer.h[1].mlp.c_fc.weight.detach()

# %%

# [d_mlp, digits (10)]
digit_sim = einsum(
    "d_mlp d_model, digits d_model -> d_mlp digits",
    W_out,
    embeds,
)

# [d_mlp]
digit_labels = digit_sim.argmax(dim=1).cpu().numpy()
digit_sim = digit_sim.cpu().numpy()

# %%

n = np.arange(10)
fourier_basis = np.column_stack(
    [
        np.ones_like(
            n
        ),  # Note that this is equal to k = 0 aka cos(2* np.pi * k * n / 10)
        np.cos(2 * np.pi * n / 5),  # k = 2
        np.sin(2 * np.pi * n / 5),  # k = 2
        np.cos(2 * np.pi * n / 10),  # k = 1
        np.sin(2 * np.pi * n / 10),  # k = 1
        (-1) ** n,  # Note that this is equal to k = 5 aka cos(2 * np.pi * k * n / 2)
        # Note that we are skipping k = 0, k = 5 for sin(2 * np.pi * k * n / 10)
        # because they end up being zero for all n in [0, 1, 2, ..., 9]
    ]
)

# Testing how well we can reconstruct our digits using the Fourier basis.
hook_modules = [
    "1.hook_resid_post",  # [batch, seq, d_model]
]
batch_size = 64

data_path = os.path.join(BASE_DIR, "data/processed_valid.txt")
with open(data_path, "r") as file_p:
    raw_data = file_p.readlines()
ci_prompts = prompt_ci_raw_format_batch(raw_data, 2, tokenizer)
ci_prompts = ci_prompts.to("cuda")
all_acts = {module_name: [] for module_name in hook_modules}
for batch_idx in range(0, len(ci_prompts), batch_size):
    curr_batch = ci_prompts[batch_idx : batch_idx + batch_size]

    with record_activations(hooked_model, hook_modules) as cache:
        _ = hooked_model(curr_batch)

    for module_name, acts in cache.items():
        all_acts[module_name].append(acts[:, -1])

all_acts = {
    module_name: torch.cat(acts, dim=0) for module_name, acts in all_acts.items()
}
last_hidden_states = all_acts["1.hook_resid_post"]
hidden_states_unemb = einsum(
    "batch d_model, vocab d_model -> batch vocab", last_hidden_states, unembeds
)
hidden_states_unemb = hidden_states_unemb.detach().cpu().numpy()

_embeds = embeds.cpu().numpy().T
_unembeds = unembeds.detach().cpu().numpy().T
_digit_sim = digit_sim

for weight_name, weight in [
    ("MLP W_out", digit_sim),
    ("embeds", _embeds),
    ("unembeds", _unembeds),
    ("last_hidden", hidden_states_unemb),
]:
    coeffs, *_ = np.linalg.lstsq(fourier_basis, weight.T, rcond=None)
    # recon: [d_mlp, 10]
    recon = (fourier_basis @ coeffs).T

    # row_var: [d_mlp, 1]
    row_var = weight.var(axis=1, keepdims=True)
    row_error = ((weight - recon) ** 2).mean(axis=1, keepdims=True)
    R2_per_row = 1 - row_error / row_var
    print(f"{weight_name}: median R^2 over rows:", np.median(R2_per_row))

# %%

# Do we need all the Fourier bases?

basis = [np.ones_like(n)]
for k in (1, 2, 3, 4):
    basis.append(np.cos(2 * np.pi * n * k / 10))
    basis.append(np.sin(2 * np.pi * n * k / 10))
basis += [(-1) ** n]

# basis: [10 (n), 10 (num bases)]
basis = np.column_stack(basis)

# %%

# Checking if digit embeddings also lie on Fourier bases.

# Try adding more Fourier bases in the following order
idx_order = [0, 9, 3, 4, 1, 2, 5, 6, 7, 8]  # DC, k=5, ±1, ±2, ±3, ±4

for weight_name, weight in [
    ("MLP W_out", digit_sim),
    ("embeds", _embeds),
    ("unembeds", _unembeds),
    ("last_hidden", hidden_states_unemb),
]:
    R2s_cumulative = []
    recon = np.zeros_like(weight)
    coeffs, *_ = np.linalg.lstsq(basis, weight.T, rcond=None)
    sst = ((weight - weight.mean(axis=1, keepdims=True)) ** 2).sum()
    for j, idx in enumerate(idx_order, 1):
        recon += np.outer(basis[:, idx], coeffs[idx]).T
        sse = ((weight - recon) ** 2).sum()
        R2 = 1 - sse / sst

        R2s_cumulative.append(R2.item())
    print(f"{weight_name}: R^2 after adding bases:", R2s_cumulative)
