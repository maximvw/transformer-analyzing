import os, re, argparse
import numpy as np
import pandas as pd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import LogNorm

plt.rcParams["text.usetex"] = True


def _load_matrix_from_csv(
    path,
    metric="handle_ratio",
):
    df = pd.read_csv(path)
    metrics_raw = df[metric].values
    handle_ratios = []
    for x in metrics_raw:
        x = x.replace("[", "").replace("]", "")
        x = x.split(",")
        x = [float(xx) for xx in x]
        handle_ratios.append(x)
    return df["step"], handle_ratios


def load_metrics(
    path,
    metric_names,
):
    df = pd.read_csv(path)
    metrics = {}
    for metric in metric_names:
        metrics_raw = df[metric].values
        metrics[metric] = []
        for x in metrics_raw:
            x = x.replace("[", "").replace("]", "")
            x = x.split(",")
            x = [float(xx) for xx in x]
            metrics[metric].append(x)
    return df["step"], metrics


def _nice_extent(steps):
    positions = list(range(0, 8))
    dx = steps[1] - steps[0]
    dy = 1.0
    return [
        steps.min() - dx / 2,
        steps.max() + dx / 2,
        0 - dy / 2,
        7 + dy / 2,
    ]


def plot_heatmaps_and_loss(
    csv_a,
    csv_b,
    out_path,
    metrics,
    figsize=(10, 6),
):
    steps_a, metrics_a = load_metrics(csv_a, metrics)
    steps_b, metrics_b = load_metrics(csv_b, metrics)

    first_metric = metrics[0]
    second_metric = metrics[1]

    handles_a = np.array(metrics_a[first_metric]).T  # (8, num_steps)
    handles_b = np.array(metrics_b[first_metric]).T  # (8, num_steps)
    loss_a = np.array(metrics_a[second_metric]).T[::-1]
    loss_b = np.array(metrics_b[second_metric]).T[::-1]

    if len(metrics) > 2:
        third_metric = metrics[2]
        g_norm_a = np.array(metrics_a[third_metric]).T[::-1]
        g_norm_b = np.array(metrics_b[third_metric]).T[::-1]
        g_norm_b = g_norm_b[:, :cutoff]

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    height_ratios = [1, 1]
    if len(metrics) > 2:
        height_ratios.append(1)
    gs = fig.add_gridspec(len(metrics), 2, height_ratios=height_ratios)

    ax_h0 = fig.add_subplot(gs[0, 0])
    vmin = np.nanmin([np.nanmin(handles_a), np.nanmin(handles_b)])
    vmax = np.nanmax([np.nanmax(handles_a), np.nanmax(handles_b)])
    norm = LogNorm(vmin=0.1, vmax=vmax)

    im0 = ax_h0.imshow(
        handles_a,
        aspect="auto",
        extent=_nice_extent(steps_a),
        norm=norm,
        interpolation="nearest",
    )
    ax_h0.set_title("(a) Standard Fine Tuning")
    ax_h0.set_xlabel("Gradient Steps (Log Scale)")
    ax_h0.set_ylabel(rf"$C_k$ Position")
    ax_h0.set_yticks(list(range(8)))
    ax_h0.set_yticklabels(
        [
            r"$C_0$",
            r"$C_1$",
            r"$C_2$",
            r"$C_3$",
            r"$C_4$",
            r"$C_5$",
            r"$C_6$",
            r"$C_7$",
        ]
    )

    ax_h1 = fig.add_subplot(gs[0, 1])
    vmin = np.nanmin(handles_b)
    vmax = 1000
    vmax = 0
    vmax = np.nanmax(handles_b)
    im1 = ax_h1.imshow(
        handles_b,
        aspect="auto",
        extent=_nice_extent(steps_b),
        norm=norm,
        interpolation="nearest",
    )
    ax_h1.set_title("(b) Training with Auxiliary Loss")
    ax_h1.set_xlabel("Gradient Steps (Log Scale)")
    ax_h1.set_ylabel(r"$C_k$ Position")
    ax_h1.set_yticks(list(range(8)))
    ax_h1.set_yticklabels(
        [
            r"$C_0$",
            r"$C_1$",
            r"$C_2$",
            r"$C_3$",
            r"$C_4$",
            r"$C_5$",
            r"$C_6$",
            r"$C_7$",
        ]
    )

    ax_h0.invert_yaxis()
    ax_h1.invert_yaxis()

    cbar2 = plt.colorbar(im1, ax=ax_h1, shrink=0.9, pad=0.1)
    cbar2.ax.set_title("Grad Norm\n(Log Scale)", fontsize=10)

    ax_c0 = fig.add_subplot(gs[1, 0], sharex=ax_h0)
    colors = [
        "red",
        "darkorange",
        "gold",
        "green",
        "blue",
        "darkviolet",
        "magenta",
        "saddlebrown",
    ]
    P, S = loss_a.shape
    for idx in range(P):
        lw = 1
        if idx < 2:
            ax_c0.plot(
                steps_a,
                loss_a[idx],
                colors[idx],
                alpha=0.8,
                linewidth=lw,
                label=rf"$c_{{{idx}}}$",
                zorder=10,
            )
        else:
            ax_c0.plot(
                steps_a,
                loss_a[idx],
                colors[idx],
                alpha=0.8,
                linewidth=lw,
                label=rf"$c_{{{idx}}}$",
                zorder=1,
            )
    ax_c0.set_ylim(top=2.5, bottom=-0.05)
    ax_c0.margins(y=0.05)
    ax_c0.set_ylabel(r"Loss (Per $c_k$ Position)")
    ax_c0.set_xlabel("Gradient Steps (Log Scale)")

    ax_c1 = fig.add_subplot(gs[1, 1], sharex=ax_h1)
    P, S = loss_b.shape
    for idx in range(P):
        if idx < 2:
            ax_c1.plot(
                steps_b,
                loss_b[idx],
                colors[idx],
                alpha=0.9,
                linewidth=lw,
                label=rf"$c_{{{idx}}}$",
                zorder=10,
            )
        else:
            ax_c1.plot(
                steps_b,
                loss_b[idx],
                colors[idx],
                alpha=0.7,
                linewidth=lw,
                label=rf"$c_{{{idx}}}$",
            )
    ax_c1.set_ylim(top=2.5, bottom=-0.05)
    ax_c1.margins(y=0.05)
    ax_c1.set_ylabel(r"Loss (Per $c_k$ Position)")
    ax_c1.set_xlabel("Gradient Steps (Log Scale)")

    ax_c0.spines["top"].set_visible(False)
    ax_c0.spines["right"].set_visible(False)
    ax_c1.spines["top"].set_visible(False)
    ax_c1.spines["right"].set_visible(False)

    for s in ax_c0.spines.values():
        s.set_zorder(0)
    for s in ax_c1.spines.values():
        s.set_zorder(0)

    leg2 = ax_c1.legend(bbox_to_anchor=(1, 1), loc="upper left", title="Position")
    leg2.set_in_layout(False)

    def custom_exp_formatter(x, pos):
        """
        Formats the tick label to display exponential notation without the '+' for positive exponents.
        """
        s = f"{x:.0e}"  # Format as scientific notation
        mantissa, exponent = s.split("e")
        exponent = int(exponent)
        if exponent >= 0:
            return f"{mantissa}e{exponent}"  # Remove '+' if exponent is positive
        else:
            return f"{mantissa}e{exponent}"

    ax_h0.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.0e"))
    ax_h1.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.0e"))
    ax_c0.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.0e"))
    ax_c1.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.0e"))

    # third metric:
    if len(metrics) > 2:
        ax_g0 = fig.add_subplot(gs[2, 0], sharex=ax_h0)
        for idx in range(P):
            if idx < 2:
                ax_g0.plot(
                    steps_a,
                    g_norm_a[idx],
                    colors[idx],
                    alpha=0.8,
                    linewidth=lw,
                    label=rf"$c_{{{idx}}}$",
                    zorder=10,
                )
            else:
                ax_g0.plot(
                    steps_a,
                    g_norm_a[idx],
                    colors[idx],
                    alpha=0.8,
                    linewidth=lw,
                    label=rf"$c_{{{idx}}}$",
                )
        ax_g0.set_ylim(top=10, bottom=-0.05)
        ax_g0.margins(y=0.05)
        ax_g0.set_ylabel(r"Gradient Norm (Per $c_k$ Position)")
        ax_g0.set_xlabel("Gradient Steps (Log Scale)")
        ax_g0.spines["top"].set_visible(False)
        ax_g0.spines["right"].set_visible(False)
        for s in ax_g0.spines.values():
            s.set_zorder(0)

        ax_g1 = fig.add_subplot(gs[2, 1], sharex=ax_h1)
        for idx in range(P):
            if idx < 2:
                ax_g1.plot(
                    steps_b,
                    g_norm_b[idx],
                    colors[idx],
                    alpha=0.8,
                    linewidth=lw,
                    label=rf"$c_{{{idx}}}$",
                    zorder=10,
                )
            else:
                ax_g1.plot(
                    steps_b,
                    g_norm_b[idx],
                    colors[idx],
                    alpha=0.8,
                    linewidth=lw,
                    label=rf"$c_{{{idx}}}$",
                )

        # ax_g1.set_ylim(top=1.5, bottom=-0.05)
        ax_g1.margins(y=0.05)
        ax_g1.set_ylabel(r"Gradient Norm (Per $c_k$ Position)")
        ax_g1.set_xlabel("Gradient Steps (Log Scale)")
        ax_g1.spines["top"].set_visible(False)
        ax_g1.spines["right"].set_visible(False)
        for s in ax_g1.spines.values():
            s.set_zorder(0)

    
    ax_h0.set_xscale("log")
    ax_h1.set_xscale("log")
    ax_c0.set_xscale("log")
    ax_c1.set_xscale("log")

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def main():
    csv_a = "ckpts/vanilla_ft/grad_probe_log.csv"
    csv_b = "ckpts/aux_head/grad_probe_log.csv"
    output_filepath = "paper_figures/grad_norms_and_losses.pdf"
    plot_heatmaps_and_loss(
        csv_a=csv_a,
        csv_b=csv_b,
        metrics=["grad_norm_per_pos", "loss_per_pos"],
        out_path=output_filepath,
    )


if __name__ == "__main__":
    main()
