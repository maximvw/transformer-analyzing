from typing import List, Tuple, Mapping
import os
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from src.model_utils import load_hf_model
from src.data_utils import (
    read_operands,
    prompt_ci_raw_format_batch,
    get_ci,
    extract_answer,
)

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

data_path = os.path.join(BASE_DIR, "data/processed_valid.txt")
with open(data_path, "r") as file_p:
    raw_data = file_p.readlines()

# [batch, seq]
ci_prompts = prompt_ci_raw_format_batch(raw_data, 2, tokenizer)
ci_prompts = ci_prompts.to("cuda")

operands = read_operands(data_path, flip_operands=True, as_int=True)


# Build labels.
digits_data = [" " + text.strip() + " " for text in raw_data]
c2s = [get_ci(x, 2) for x in digits_data]
texts = tokenizer.batch_decode(ci_prompts)

labels = {
    "digit": c2s,
}

# %%


hook_modules = [
    # "1.mlp.hook_mlp_mid",  # [batch, seq, d_mlp]
    # "1.mlp",  # [batch, seq, d_model]
    "1.hook_resid_post",  # [batch, seq, d_model]
    # "base_model.lm_head",
]
batch_size = 64

all_acts = {module_name: [] for module_name in hook_modules}
for batch_idx in range(0, len(ci_prompts), batch_size):
    curr_batch = ci_prompts[batch_idx : batch_idx + batch_size]

    with record_activations(hooked_model, hook_modules) as cache:
        _ = hooked_model(curr_batch)

    for module_name, acts in cache.items():
        if "attn.hook_attn_pattern" in module_name:
            all_acts[module_name].append(acts.cpu())
        else:
            # Index last timestep.
            all_acts[module_name].append(acts[:, -1].cpu())

all_acts = {
    module_name: torch.cat(acts, dim=0) for module_name, acts in all_acts.items()
}
if "base_model.lm_head" in all_acts:
    digits = [f" {i}" for i in range(10)]
    digit_token_ids = tokenizer(
        digits, return_tensors="pt", padding=False, truncation=True
    ).input_ids.squeeze()
    all_acts["base_model.lm_head"] = all_acts["base_model.lm_head"][:, digit_token_ids]

# %%

output_dir = os.path.join(
    BASE_DIR,
    f"paper_figures",
)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# %%


@torch.no_grad()
def plot_pca(
    acts: torch.Tensor,
    labels: Mapping[str, List[str]],
    head_idx=None,
    output_dir=None,
):
    """
    acts: [batch, d]
    """
    print(f"   plot_pca: acts.shape: {acts.shape}")
    assert len(acts.shape) == 2
    pca = PCA(n_components=3)
    reduce_out = pca.fit_transform(acts)

    # df for plotly
    df = pd.DataFrame(
        {
            "PC1": reduce_out[:, 0],
            "PC2": reduce_out[:, 1],
            "PC3": reduce_out[:, 2],
            **labels,
        }
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=df["PC1"],
            y=df["PC2"],
            z=df["PC3"],
            mode="markers",
            marker=dict(
                color=df["digit"],
                colorscale="viridis",
                showscale=False,
                size=2,
            ),
            name="digits",
            showlegend=False,
        )
    )

    centroids = df.groupby("digit").mean().sort_index()

    def order_by_angle(idxs):
        xy = centroids.loc[idxs, ["PC2", "PC3"]].to_numpy()
        ang = np.arctan2(xy[:, 1], xy[:, 0])
        return [idx for _, idx in sorted(zip(ang, idxs))]

    even_digits = [d for d in centroids.index if d % 2 == 0]
    odd_digits = [d for d in centroids.index if d % 2 == 1]

    even_order = order_by_angle(even_digits)
    odd_order = order_by_angle(odd_digits)

    even_loop = even_order + [even_order[0]]
    odd_loop = odd_order + [odd_order[0]]

    def add_loop_trace(name, idxs, line_width=5, dash=None):
        pts = centroids.loc[idxs]
        fig.add_trace(
            go.Scatter3d(
                x=pts["PC1"],
                y=pts["PC2"],
                z=pts["PC3"],
                mode="lines+markers",
                line=dict(width=line_width, dash=dash, color="black"),
                marker=dict(size=2),
                name=name,
                showlegend=False,
            )
        )

    # ---- Add the two pentagon loops (centroids of evens/odds) --------------
    add_loop_trace("even pentagon (centroids)", even_loop, line_width=3.5)
    add_loop_trace("odd pentagon (centroids)", odd_loop, line_width=3.5)

    # ---- Add the vertical prism edges: connect n -> n+5 --------------------
    for n in range(5):
        a = centroids.loc[n]
        b = centroids.loc[n + 5]
        fig.add_trace(
            go.Scatter3d(
                x=[a["PC1"], b["PC1"]],
                y=[a["PC2"], b["PC2"]],
                z=[a["PC3"], b["PC3"]],
                mode="lines",
                line=dict(width=3.5, color="black"),
                name=f"edge {n}↔{n+5}",
                showlegend=False,
            )
        )

    # ---- Annotate each cluster with its digit at the centroid ----------
    labels_df = centroids.reset_index()  # columns: digit, PC1, PC2, PC3

    label_nudge = 0.05
    y_range = df["PC2"].max() - df["PC2"].min()
    z_range = df["PC3"].max() - df["PC3"].min()
    dy = label_nudge * y_range
    dz = label_nudge * z_range

    x_lab = labels_df["PC1"].to_numpy().copy()
    y_lab = labels_df["PC2"].to_numpy().copy()
    z_lab = labels_df["PC3"].to_numpy().copy()

    orientation_map = {
        0: ["up"],
        1: ["left"],
        2: ["down", "right"],
        3: ["down"],
        4: ["up"],
        5: ["left"],
        6: ["right"],
        7: ["down"],
        8: ["right"],
        9: ["up"],
    }
    weight = {
        0: 1.1,
        1: 1.5,
        2: 1.5,
        3: 0.5,
        4: 1.1,
        5: 1,
        6: 1,
        7: 2,
        8: 1,
        9: 0.5,
    }
    for i, d in enumerate(labels_df["digit"].to_numpy()):
        ori = orientation_map[int(d)]
        if "up" in ori:
            z_lab[i] += weight[int(d)] * dz
        if "down" in ori:
            z_lab[i] -= weight[int(d)] * dz
        if "left" in ori:
            y_lab[i] -= weight[int(d)] * dy
        if "right" in ori:
            y_lab[i] += weight[int(d)] * dy

    annos = []
    for i, d in enumerate(labels_df["digit"].to_numpy()):
        annos.append(
            dict(
                x=float(x_lab[i]),
                y=float(y_lab[i]),
                z=float(z_lab[i]),
                text=str(int(d)),
                showarrow=False,
                font=dict(size=16, color="black"),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
                opacity=1.0,
            )
        )

    # Apply to the 3D scene (this draws above all traces)
    fig.update_layout(scene=dict(annotations=annos))

    def arc(digits, n_samples=80, pc1=None, scale=1.0):
        pts = centroids.loc[digits, ["PC1", "PC2", "PC3"]].to_numpy()
        if pc1 is None:
            pc1 = pts[:, 0].mean() - 5

        ang = np.arctan2(pts[:, 2], pts[:, 1])
        rad = np.linalg.norm(pts[:, 1:3], axis=1).mean()

        unwrapped = [ang[0]]
        for a in ang[1:]:
            u = a
            while u > unwrapped[-1]:
                u -= 2 * np.pi
            unwrapped.append(u)
        thetas = np.linspace(unwrapped[0], unwrapped[-1], n_samples)

        x = np.full(n_samples, pc1)
        y = 1.4 * scale * rad * np.cos(thetas)
        z = 1.4 * scale * rad * np.sin(thetas)
        return x, y, z

    x_arc, y_arc, z_arc = arc([0, 4, 8], n_samples=80)

    fig.add_trace(
        go.Scatter3d(
            x=x_arc,
            y=y_arc,
            z=z_arc,
            mode="lines",
            line=dict(width=3, color="black", dash="longdash"),
            showlegend=False,
        )
    )

    # arrowhead as a small cone at the end of the arc (points in the tangent direction)
    vx = x_arc[-1] - x_arc[-2]
    vy = y_arc[-1] - y_arc[-2]
    vz = z_arc[-1] - z_arc[-2]
    fig.add_trace(
        go.Cone(
            x=[x_arc[-1]],
            y=[y_arc[-1]],
            z=[z_arc[-1]],
            u=[vx],
            v=[vy],
            w=[vz],
            anchor="tip",
            sizemode="absolute",
            sizeref=3.0,
            colorscale=[[0, "black"], [1, "black"]],
            showscale=False,
            hoverinfo="skip",
        )
    )

    # label_text = r"$\cos,\ \sin\!\left(\tfrac{2\pi n}{5}\right)$<br>$(k=2)$"
    label_text = "cos, sin(2\u03C0 n / 5)<br>(k=2)"

    # text label near the middle of the arc
    mid = len(x_arc) // 2
    fig.update_layout(
        scene=dict(
            annotations=list(fig.layout.scene.annotations)
            + [
                dict(
                    x=float(x_arc[mid] * 1.2),
                    y=float(y_arc[mid] * 1.06),  # nudge out a bit
                    z=float(z_arc[mid] * 1.25),
                    text=label_text,
                    showarrow=False,
                    font=dict(size=13, color="black"),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1,
                    align="center",
                )
            ]
        )
    )

    # ---------- Bidirectional arrow between digits 0 and 5 (±5) ----------
    a = centroids.loc[0, ["PC1", "PC2", "PC3"]].to_numpy()
    b = centroids.loc[5, ["PC1", "PC2", "PC3"]].to_numpy()
    a[0] += 0.1 * (df["PC1"].max() - df["PC1"].min())
    b[0] += 0.1 * (df["PC1"].max() - df["PC1"].min())

    a[1] -= 0.18 * y_range
    b[1] -= 0.18 * y_range

    a[2] += 0.18 * z_range
    b[2] += 0.18 * z_range

    # the connecting line
    fig.add_trace(
        go.Scatter3d(
            x=[a[0], b[0]],
            y=[a[1], b[1]],
            z=[a[2], b[2]],
            mode="lines",
            line=dict(width=3, color="black", dash="longdash"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # arrowheads at both ends (cones pointing outward)
    d = b - a
    fig.add_trace(
        go.Cone(
            x=[b[0]],
            y=[b[1]],
            z=[b[2]],
            u=[d[0]],
            v=[d[1]],
            w=[d[2]],
            anchor="tip",
            sizemode="absolute",
            sizeref=3,
            colorscale=[[0, "black"], [1, "black"]],
            showscale=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Cone(
            x=[a[0]],
            y=[a[1]],
            z=[a[2]],
            u=[-d[0]],
            v=[-d[1]],
            w=[-d[2]],
            anchor="tip",
            sizemode="absolute",
            sizeref=3,
            colorscale=[[0, "black"], [1, "black"]],
            showscale=False,
            hoverinfo="skip",
        )
    )

    # label slightly offset from the midpoint
    midpt = 0.5 * (a + b)
    x_range = df["PC1"].max() - df["PC1"].min()
    y_range = df["PC2"].max() - df["PC2"].min()
    z_range = df["PC3"].max() - df["PC3"].min()
    fig.update_layout(
        scene=dict(
            annotations=list(fig.layout.scene.annotations)
            + [
                dict(
                    x=float(midpt[0] + 0.15 * x_range),
                    y=float(midpt[1] + 0.15 * y_range),
                    z=float(midpt[2] + 0.25 * z_range),
                    text="Parity (±5)<br>(k=5)",
                    showarrow=False,
                    font=dict(size=13, color="black"),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1,
                )
            ]
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                # backgroundcolor="rgba(255,255,255,0)",
                title="PC1",
                showticklabels=False,
                ticks="",
                showgrid=True,
            ),
            yaxis=dict(
                # backgroundcolor="rgba(0,0,0,0)",
                title="PC2",
                showticklabels=False,
                ticks="",
                # showgrid=False,
            ),
            zaxis=dict(
                # backgroundcolor="rgba(0,0,0,0)",
                title="PC3",
                showticklabels=False,
                ticks="",
                # showgrid=False,
            ),
        ),
        margin=dict(l=40, r=40, t=80, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_scenes(aspectmode="data")

    fig.show()
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        out_filepath = f"{output_dir}/fourier_basis.pdf"
        fig.write_image(out_filepath)


# %%

import plotly.io as pio

pio.renderers.default = "iframe"
plot_pca(
    all_acts["1.hook_resid_post"],
    labels=labels,
    output_dir=output_dir,
)
