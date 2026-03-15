from typing import List, Tuple, Mapping
import os
import numpy as np
import torch
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from plotly.subplots import make_subplots

from src.model_utils import load_hf_model
from src.data_utils import prompt_ci_raw_format_batch, get_ith_a_or_b_digit
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

data_path = os.path.join(BASE_DIR, "data/processed_valid_large.txt")
with open(data_path, "r") as file_p:
    raw_data = file_p.readlines()

# [batch, seq]
raw_data = [x.split("||")[0].strip() for x in raw_data]
ci_prompts = prompt_ci_raw_format_batch(raw_data, 2, tokenizer)
ci_prompts = ci_prompts.to("cuda")

# Build labels.

digits_data = [" " + text.strip() + " " for text in raw_data]

a1s = [get_ith_a_or_b_digit(x, "a", 1) for x in digits_data]
b1s = [get_ith_a_or_b_digit(x, "b", 1) for x in digits_data]

# %%

labels = {
    "a1s": a1s,
    "b1s": b1s,
}

# %%


hook_modules = [
    "0.attn.hook_attn_output_per_head",  # [batch, seq, head_idx, d_model]
]
batch_size = 64

all_acts = {module_name: [] for module_name in hook_modules}
for batch_idx in range(0, len(ci_prompts), batch_size):
    curr_batch = ci_prompts[batch_idx : batch_idx + batch_size]

    with record_activations(hooked_model, hook_modules) as cache:
        _ = hooked_model(curr_batch)

    for module_name, acts in cache.items():
        # Index last timestep.
        all_acts[module_name].append(acts[:, -1].cpu())

acts = {module_name: torch.cat(acts, dim=0) for module_name, acts in all_acts.items()}
acts_per_head = acts["0.attn.hook_attn_output_per_head"]


# %%

# Get minkowski data
ci_prompts = prompt_ci_raw_format_batch(raw_data, 0, tokenizer)
ci_prompts = ci_prompts.to("cuda")
b0s = [get_ith_a_or_b_digit(x, "b", 0) for x in digits_data]
hook_modules = [
    "0.attn",  # [batch, seq, d_model]
]
batch_size = 64
labels_overall = {"b0s": b0s}

all_acts = {module_name: [] for module_name in hook_modules}
for batch_idx in range(0, len(ci_prompts), batch_size):
    curr_batch = ci_prompts[batch_idx : batch_idx + batch_size]

    with record_activations(hooked_model, hook_modules) as cache:
        _ = hooked_model(curr_batch)

    for module_name, acts in cache.items():
        # Index last timestep.
        all_acts[module_name].append(acts[:, -1].cpu())

acts = {module_name: torch.cat(acts, dim=0) for module_name, acts in all_acts.items()}[
    "0.attn"
]


# %%

output_dir = os.path.join(BASE_DIR, f"paper_figures")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# %%


@torch.no_grad()
def plot_pca(
    acts_per_head: torch.Tensor,
    acts_overall,
    labels: Mapping[str, List[str]],
    labels_overall,
    head_idx,
    box_value,
    output_dir=None,
):
    """
    acts_per_head: [batch, d]
    """
    print(f"   plot_pca: acts_per_head.shape: {acts_per_head.shape}")
    assert len(acts_per_head.shape) == 2
    pca = PCA(n_components=3)
    reduce_out = pca.fit_transform(acts_per_head)

    # df for plotly
    df = pd.DataFrame(
        {
            "PC1": reduce_out[:, 0],
            "PC2": reduce_out[:, 1],
            "PC3": reduce_out[:, 2],
            **labels,
        }
    )
    reduce_out_overall = PCA(n_components=3).fit_transform(acts_overall)
    df_overall = pd.DataFrame(
        {
            "PC1": reduce_out_overall[:, 0],
            "PC2": reduce_out_overall[:, 1],
            "PC3": reduce_out_overall[:, 2],
            **labels_overall,
        }
    )

    fig = make_subplots(
        rows=1,
        cols=4,
        specs=[
            [{"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}]
        ],
        subplot_titles=(
            "(a) Attention Layer 1<br>Colored by b0<br> ",
            "(b) Attention Layer 1 Head 3<br>Colored by b1<br> ",
            "(c) Attention Layer 1 Head 3<br>Colored by a1<br> ",
            "(d) Attention Layer 1 Head 3<br>Zoomed in: Points inside box<br>Colored by a1",
        ),
        horizontal_spacing=0.03,
    )

    fig.add_trace(
        go.Scatter3d(
            x=df_overall["PC1"],
            y=df_overall["PC2"],
            z=df_overall["PC3"],
            mode="markers",
            marker=dict(
                color=df_overall["b0s"],
                colorscale="viridis",
                showscale=False,
                size=2,
            ),
            visible=True,
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter3d(
            x=df["PC1"],
            y=df["PC2"],
            z=df["PC3"],
            mode="markers",
            marker=dict(
                color=df["b1s"],
                colorscale="viridis",
                showscale=True,
                size=2,
            ),
            visible=True,
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter3d(
            x=df["PC1"],
            y=df["PC2"],
            z=df["PC3"],
            mode="markers",
            marker=dict(
                color=df["a1s"],
                colorscale="viridis",
                showscale=True,
                size=2,
            ),
            visible=True,  # show by default
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    box_col = "b1s"
    m_cluster = (df[box_col] == box_value).to_numpy()
    pts = df.loc[m_cluster, ["PC1", "PC2", "PC3"]].to_numpy()

    # axis-aligned bounding box with a small padding
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    rng = maxs - mins
    # box_pad_frac = 0.001
    box_pad_frac = 0.02
    mins = mins - box_pad_frac * rng
    maxs = maxs + box_pad_frac * rng

    (xmin, ymin, zmin), (xmax, ymax, zmax) = mins, maxs
    # xmax = xmax - 0.1 * (xmax - xmin)
    # zmax = zmax - 0.1 * (zmax - zmin)
    corners = np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmin, ymax, zmin],
            [xmax, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmin, ymax, zmax],
            [xmax, ymax, zmax],
        ]
    )

    # edges as pairs of indices
    edges = [
        (0, 1),
        (1, 3),
        (3, 2),
        (2, 0),
        (4, 5),
        (5, 7),
        (7, 6),
        (6, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    x_lines, y_lines, z_lines = [], [], []
    for i, j in edges:
        x_lines += [corners[i, 0], corners[j, 0], None]
        y_lines += [corners[i, 1], corners[j, 1], None]
        z_lines += [corners[i, 2], corners[j, 2], None]

    fig.add_trace(
        go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode="lines",
            line=dict(width=5),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    inside = (
        (df["PC1"].between(xmin, xmax))
        & (df["PC2"].between(ymin, ymax))
        & (df["PC3"].between(zmin, zmax))
    ).to_numpy()

    fig.add_trace(
        go.Scatter3d(
            x=df.loc[inside, "PC1"],
            y=df.loc[inside, "PC2"],
            z=df.loc[inside, "PC3"],
            mode="markers",
            marker=dict(
                color=df.loc[inside, "a1s"],
                colorscale="viridis",
                showscale=True,
                size=3.5,
            ),
            visible=True,
            showlegend=False,
        ),
        row=1,
        col=4,
    )

    fig.update_layout(
        margin=dict(l=30, r=30, t=70, b=30),
        height=600,
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
        scene2=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
        scene3=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
        margin=dict(l=40, r=40, t=80, b=40),
    )

    # ---- sizing helpers for LaTeX ----
    width_in = 7.2
    height_in = 0.33 * width_in
    dpi = 300

    width_px = int(round(width_in * dpi))
    height_px = int(round(height_in * dpi))

    fig.update_layout(
        width=width_px,
        height=height_px,
    )
    axis_title_size = 20
    for col in range(1, 5):
        fig.update_scenes(
            xaxis=dict(
                title=dict(text=f"PC1", font=dict(size=axis_title_size)),
                showticklabels=False,
                ticks="",
                showbackground=True,
            ),
            yaxis=dict(
                title=dict(text=f"PC2", font=dict(size=axis_title_size)),
                showticklabels=False,
                ticks="",
                showbackground=True,
            ),
            zaxis=dict(
                title=dict(text=f"PC3", font=dict(size=axis_title_size)),
                showticklabels=False,
                ticks="",
                showbackground=True,
            ),
            aspectmode="cube",
            row=1,
            col=col,
        )

    subplot_title_size = 30
    gap_px = -48
    if fig.layout.annotations:
        for i, ann in enumerate(fig.layout.annotations):
            ann.font = dict(size=subplot_title_size)
            scene = getattr(fig.layout, f"scene{i+1}")
            domx = scene.domain.x
            domy = scene.domain.y
            ann.update(
                x=(domx[0] + domx[1]) / 2,
                y=domy[1],
                xref="paper",
                yref="paper",
                xanchor="center",
                yanchor="bottom",
                yshift=gap_px,
                font=dict(size=subplot_title_size),
                showarrow=False,
            )
    fig.update_layout(showlegend=False)
    fig.for_each_trace(lambda t: t.update(marker=dict(showscale=False)))
    fig.update_layout(coloraxis_showscale=False)

    sL = getattr(fig.layout, "scene")
    sR = getattr(fig.layout, f"scene2")
    x_sep = (sL.domain.x[1] + sR.domain.x[0]) / 2
    y0 = max(sL.domain.y[0], sR.domain.y[0])
    y1 = min(sL.domain.y[1], sR.domain.y[1])
    fig.add_shape(
        type="line",
        x0=x_sep,
        x1=x_sep,
        y0=y0,
        y1=y1,
        xref="paper",
        yref="paper",
        line=dict(color="black", width=4),
        layer="above"
    )

    # fig.update_scenes(aspectmode="data")
    fig.show()
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        out_filepath = f"{output_dir}/attn_3d_pcas.pdf"
        fig.write_image(out_filepath, width=width_px, height=height_px)


# %%

for head_idx in [2]:
    _acts = acts_per_head.cpu().detach().numpy()
    _acts = _acts[:, head_idx, ...]
    acts_overall = acts.cpu().detach().numpy()
    for box_value in [3]:
        plot_pca(
            _acts,
            acts_overall,
            labels,
            labels_overall,
            head_idx=head_idx,
            box_value=box_value,
            output_dir=output_dir,
        )

