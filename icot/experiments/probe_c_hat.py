import os
import sys
import torch
import numpy as np
import importlib
import json
import time
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from src.model_utils import load_hf_model, load_c_hat_model
from src.data_utils import prompt_ci_operands
from src.probes import RegressionProbe
from src.HookedModel import convert_to_hooked_model
from src.ActivationCache import record_activations
from constants import BASE_DIR


plt.rcParams["text.usetex"] = True

hook_modules = [
    "0.hook_resid_mid",
    "0.hook_resid_post",
    "1.hook_resid_mid",
    "1.hook_resid_post",
]

# load model and tokenizer
model_path = os.path.join(BASE_DIR, "ckpts/2L4H/")
config_path = os.path.join(model_path, "config.json")
state_dict_path = os.path.join(model_path, "state_dict.bin")

hooked_model, tokenizer = load_hf_model(config_path, state_dict_path)
hooked_model.to("cuda").eval()
convert_to_hooked_model(hooked_model)

sft_model, _ = load_c_hat_model(os.path.join(BASE_DIR, "ckpts/vanilla_ft/ckpt.pt"))
sft_model.to("cuda")

data_path = os.path.join(BASE_DIR, "data/processed_valid.txt")

with open(data_path, "r") as f:
    texts = f.readlines()

texts = [
    text.replace(" ", "").replace("\n", "").split("||")[0].split("*")
    for text in texts
    if text != "\n"
]

operands = [(int(a[::-1]), int(b[::-1])) for a, b in texts]

prompt_text, tokens = prompt_ci_operands(operands, 8, tokenizer, device="cuda")


def get_c_hats(a, b):
    c_hats = []
    carrys = []
    pair_sums = []
    a_digits = [int(d) for d in str(a)[::-1]]
    b_digits = [int(d) for d in str(b)[::-1]]
    total_len = len(a_digits) + len(b_digits)

    for ii in range(total_len):
        aibi_sum = 0
        # sum products along the "diagonal" ii
        for a_ii in range(ii, -1, -1):
            b_ii = ii - a_ii
            if 0 <= a_ii < len(a_digits) and 0 <= b_ii < len(b_digits):
                aibi_sum += a_digits[a_ii] * b_digits[b_ii]

        pair_sums.append(aibi_sum)

        # add carry from previous running sum
        if len(c_hats) > 0:
            aibi_sum += c_hats[-1] // 10

        c_hats.append(aibi_sum)
        carrys.append(aibi_sum // 10)

    return c_hats, carrys, pair_sums


labels = []
for a, b in operands:
    c_hats, carrys, pair_sums = get_c_hats(a, b)
    labels.append(c_hats)

labels = torch.tensor(labels, dtype=torch.float32)

# shuffle and split
torch.manual_seed(123)
shuffle_idx = torch.randperm(len(tokens))
tokens = tokens[shuffle_idx]
labels = labels[shuffle_idx]

val_size = 1024
val_tokens = tokens[-val_size:].to("cuda")
val_labels = labels[-val_size:].to("cuda")

# record validation activations
hook_modules = [
    "0.hook_resid_mid",
    "0.hook_resid_post",
    "1.hook_resid_mid",
    "1.hook_resid_post",
]
with torch.no_grad():
    with record_activations(hooked_model, hook_modules) as cache:
        _ = hooked_model(val_tokens)

val_acts = torch.stack(
    [cache[m][:, -val_labels.shape[1] :] for m in hook_modules],
    dim=0,
)

probe_path = os.path.join(
    BASE_DIR,
    "ckpts/icot_c_hat_probe/probe.pth",
)
num_modules, val_size, seq, d_model = val_acts.shape
probe_shape = (num_modules, seq, d_model, 1)
probe = RegressionProbe(probe_shape, 1e-3)
probe.load_weights(probe_path)

sft_probe_path = os.path.join(
    BASE_DIR,
    "ckpts/sft_c_hat_probe/probe.pth",
)
sft_probe = RegressionProbe(probe_shape, 1e-3)
sft_probe.load_weights(sft_probe_path)


with torch.no_grad():
    icot_val_preds = probe(val_acts)
    sft_val_preds = sft_probe(val_acts)

icot_metrics = probe.evaluate_probe(val_acts, val_labels)
icot_metrics = icot_metrics[-1][2]  # (8,)
sft_metrics = sft_probe.evaluate_probe(val_acts, val_labels)
sft_metrics = sft_metrics[-1][2]  # (8,)

val_labels = val_labels.cpu().numpy()
icot_val_preds = icot_val_preds.cpu().numpy()
sft_val_preds = sft_val_preds.cpu().numpy()

n_rows = 2
n_cols = 5
fig, axes = plt.subplots(
    n_rows, n_cols, figsize=(12, 5), gridspec_kw={"hspace": 0.45}
)

for row in range(n_rows):
    if row == 0:
        probe_preds = sft_val_preds
        metrics = sft_metrics
    else:
        probe_preds = icot_val_preds
        metrics = icot_metrics

    for col_idx, c_i in enumerate(range(2, 7)):
        ax = axes[row, col_idx]
        _val_labels = val_labels[:, c_i]
        _val_preds = probe_preds[2, :, c_i]

        min_val = min(_val_labels.min(), _val_preds.min())
        max_val = max(_val_labels.max(), _val_preds.max())
        diagonal_line = np.linspace(min_val, max_val, 100)

        sorted_indices = np.argsort(_val_labels)
        sorted_labels = _val_labels[sorted_indices]
        # val_preds: [n_layers, n_samples, n_positions]
        sorted_preds = _val_preds[sorted_indices]

        mae = metrics[c_i]

        ax.plot(
            diagonal_line,
            diagonal_line,
            "r--",
            alpha=0.7,
            linewidth=2,
            label="Perfect predictions",
        )

        ax.scatter(
            sorted_labels,
            sorted_preds,
            alpha=0.5,
            s=5,
            label="Predictions",
            color="blue",
        )
        ax.set_title(rf"$\hat{{C}}_{c_i}$ (MAE {mae:.2f})", fontsize=12)

        if row == 1:
            ax.set_xlabel(rf"True $\hat{{C}}_{c_i}$", fontsize=12)

        if col_idx == 0:
            ax.set_ylabel(rf"Predicted $\hat{{C}}$", fontsize=12)
            if row == 0:
                ax.legend(fontsize=10)
        ax.set_aspect("equal", adjustable="box")

fig.add_artist(
    plt.Line2D(
        [0.05, 0.95],
        [0.51, 0.51],
        transform=fig.transFigure,
        color="k",
        linestyle="--",
        linewidth=1,
    )
)

# Row titles
fig.text(0.5, 0.95, "SFT", ha="center", va="center", fontsize=13, fontweight="bold")
fig.text(0.5, 0.49, "ICoT", ha="center", va="center", fontsize=13, fontweight="bold")

plt.savefig("paper_figures/c_hat_probe.pdf", dpi=300, bbox_inches="tight")
