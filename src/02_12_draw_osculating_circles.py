from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from tqdm import tqdm

import scienceplots

plt.style.use("science")


from utils import pickle_load, compute_osculating_circle, compute_curvature

from parameters import WINDOW_SIZE


if __name__ == "__main__":
    meta_trajectories = pickle_load(
        "../data/intermediate/02_07_meta_trajectories_diamor.pkl"
    )

    for day in ["06", "08"]:

        for source, sink in tqdm(meta_trajectories[day].keys()):

            sc_curvature = None
            sc_velocity = None

            fig, ax = plt.subplots(figsize=(10, 5))

            # Colormap for reference
            cmap_curvature = plt.get_cmap("viridis")

            for size, m in zip([1], ["o"]):

                if size not in meta_trajectories[day][(source, sink)]:
                    continue

                meta_trajectory_size = meta_trajectories[day][(source, sink)][size]
                centers, radii = compute_osculating_circle(
                    meta_trajectory_size, WINDOW_SIZE
                )
                _, curvature = compute_curvature(meta_trajectory_size, WINDOW_SIZE)

                curvature = np.abs(curvature) * 1000

                sc_curvature = ax.scatter(
                    meta_trajectory_size[:, 1] / 1000,
                    meta_trajectory_size[:, 2] / 1000,
                    label="Dyad" if size == 2 else "Individual",
                    c=curvature,
                    marker=m,
                    s=10,
                    cmap=cmap_curvature,
                    vmin=0,
                    vmax=1,
                    alpha=0.5,
                )

                for center, radius in zip(centers, radii):
                    if np.abs(radius) > 5000:
                        continue
                    circle = plt.Circle(
                        (center[0] / 1000, center[1] / 1000),
                        radius / 1000,
                        color="black",
                        fill=False,
                        alpha=0.5,
                    )
                    ax.add_artist(circle)

            divider_curvature = make_axes_locatable(ax)
            cax_curvature = divider_curvature.append_axes("right", size="2%", pad=0.05)
            fig.colorbar(sc_curvature, cax=cax_curvature, label="$\\kappa$ [1/m]")

            curvature_legend_color = cmap_curvature(0.5)

            custom_legend = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=curvature_legend_color,
                    markersize=8,
                    label="Individual",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor=curvature_legend_color,
                    markersize=8,
                    label="Dyad",
                ),
            ]
            ax.legend(handles=custom_legend, loc="upper right")

            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_aspect("equal")
            ax.set_title("(a)")

            plt.tight_layout()
            plt.savefig(
                f"../data/figures/02_12_osculating_circles/{day}_{source}_{sink}.pdf"
            )
            plt.close()
