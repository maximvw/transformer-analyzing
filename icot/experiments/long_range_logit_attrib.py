from typing import List, Tuple
import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from src.model_utils import load_hf_model, load_c_hat_model
from src.data_utils import (
    read_operands,
    prompt_ci_raw_format_batch,
)
from src.HookedModel import convert_to_hooked_model
from src.ActivationCache import record_activations
from constants import BASE_DIR

random.seed(99)

# %%

model_name = "2L4H"
model_path = os.path.join(BASE_DIR, "ckpts/2L4H")

config_path = os.path.join(model_path, "config.json")
state_dict_path = os.path.join(model_path, "state_dict.bin")

# %%

hooked_model, tokenizer = load_hf_model(config_path, state_dict_path)
hooked_model.to("cuda")
convert_to_hooked_model(hooked_model)

sft_model, _ = load_c_hat_model(os.path.join(BASE_DIR, "ckpts/vanilla_ft/ckpt.pt"))
sft_model.to("cuda")

# %%

data_path = os.path.join(BASE_DIR, "data/processed_valid.txt")
with open(data_path, "r") as file_p:
    raw_data = file_p.readlines()


operands = read_operands(data_path, flip_operands=True, as_int=True)


# %%


def build_counter(operands, digit):
    op = operands[0]
    untouched = operands[1]
    if digit.startswith("b"):
        op = operands[1]
        untouched = operands[0]
    counter_idx = 3 - int(digit[1])

    original_digit = int(str(op)[counter_idx])
    cand = [x for x in range(10) if x != original_digit]
    if counter_idx == 0:
        cand = [x for x in range(1, 10) if x != original_digit]
    replacement = random.choice(cand)
    new_op = list(str(op))
    new_op[counter_idx] = str(replacement)
    new_op = new_op
    new_op = int("".join(new_op))
    if digit.startswith("b"):
        return (untouched, new_op)
    return (new_op, untouched)


def build_prompts(operands, ci, tokenizer):
    _formatted_data = [
        " " + " ".join(str(op[0]))[::-1] + " * " + " ".join(str(op[1]))[::-1] + " "
        for op in operands
    ]
    return prompt_ci_raw_format_batch(_formatted_data, ci, tokenizer)


icot_results = {}
sft_results = {}
for ci in range(0, 8):
    for counter_digit in ["a0", "a1", "a2", "a3", "b0", "b1", "b2", "b3"]:
        counters = [build_counter(op, counter_digit) for op in operands]
        ci_prompts = build_prompts(operands, ci, tokenizer).to("cuda")
        counter_prompts = build_prompts(counters, ci, tokenizer).to("cuda")
        counter_prompts[:, counter_prompts.shape[1] - ci :] = ci_prompts[
            :, counter_prompts.shape[1] - ci :
        ]

        # %%

        batch_size = 64
        icot_logit_diffs = []
        sft_logit_diffs = []
        for batch_idx in range(0, len(ci_prompts), batch_size):
            curr_batch = ci_prompts[batch_idx : batch_idx + batch_size]
            counter_batch = counter_prompts[batch_idx : batch_idx + batch_size]

            with torch.no_grad():
                # [batch, seq, v]
                icot_logits = hooked_model(curr_batch).logits
                sft_logits = sft_model(curr_batch).logits

            icot_orig_preds = icot_logits[:, -1].argmax(-1)
            icot_orig_logits = icot_logits[:, -1][
                torch.arange(icot_logits.shape[0]), icot_orig_preds
            ]

            sft_orig_preds = sft_logits[:, -1].argmax(-1)
            sft_orig_logits = sft_logits[:, -1][
                torch.arange(sft_logits.shape[0]), sft_orig_preds
            ]

            with torch.no_grad():
                # [batch, seq, v]
                icot_counter_logits = hooked_model(counter_batch).logits
                sft_counter_logits = sft_model(counter_batch).logits

            icot_counter_logits = icot_counter_logits[:, -1][
                torch.arange(icot_counter_logits.shape[0]), icot_orig_preds
            ]
            sft_counter_logits = sft_counter_logits[:, -1][
                torch.arange(sft_counter_logits.shape[0]), sft_orig_preds
            ]
            icot_logit_diffs.append(icot_orig_logits - icot_counter_logits)
            sft_logit_diffs.append(sft_orig_logits - sft_counter_logits)

        icot_logit_diffs = torch.cat(icot_logit_diffs, dim=0).cpu().numpy()
        sft_logit_diffs = torch.cat(sft_logit_diffs, dim=0).cpu().numpy()

        icot_logit_diffs = np.mean(icot_logit_diffs)
        sft_logit_diffs = np.mean(sft_logit_diffs)
        icot_results[(counter_digit, f"c{ci}")] = icot_logit_diffs.item()
        sft_results[(counter_digit, f"c{ci}")] = sft_logit_diffs.item()


rows = ["a3", "a2", "a1", "a0", "b3", "b2", "b1", "b0"]
cols = [f"c{k}" for k in range(8)]


def make_matrix(contribs, rows, cols):
    M = np.zeros((len(rows), len(cols)))
    for (src, tgt), val in contribs.items():
        if src in rows and tgt in cols:
            i = rows.index(src)
            j = cols.index(tgt)
            M[i, j] = val
    return M


M_icot = make_matrix(icot_results, rows, cols)
M_sft = make_matrix(sft_results, rows, cols)

vmax = max(np.abs(M_icot).max(), np.abs(M_sft).max(), 1.0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, constrained_layout=True)

for ax, M, title in zip(axes, [M_sft, M_icot], ["SFT", "ICoT"]):
    im = ax.imshow(
        M,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0,
        vmax=vmax,
    )
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=18)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=18)
    ax.set_title(title, fontsize=18)
    ax.axhline(3.5, color="k", linestyle="--", linewidth=1)

cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9, location="right")
cbar.set_label("Δ logit", fontsize=18)

plt.savefig("paper_figures/long_term_effects_heatmap.pdf", dpi=300)
