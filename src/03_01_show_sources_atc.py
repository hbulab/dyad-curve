import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from utils import pickle_load, filter_groups_by_day
from parameters import SOURCES_ATC


if __name__ == "__main__":

    groups = pickle_load("../data/intermediate/01_01_groups_atc.pkl")

    mins_x, mins_y, maxs_x, maxs_y = [], [], [], []

    for group in groups:
        trajectory = group["center_of_mass"]
        if len(trajectory) <= 0:
            continue
        mins_x.append(np.min(trajectory[:, 1]))
        maxs_x.append(np.max(trajectory[:, 1]))
        mins_y.append(np.min(trajectory[:, 2]))
        maxs_y.append(np.min(trajectory[:, 2]))

    min_x = np.min(mins_x)
    max_x = np.max(maxs_x)
    min_y = np.min(mins_y)
    max_y = np.max(maxs_y)

    CELL_SIZE = 100

    n_bin_x = int(np.ceil((max_x - min_x) / CELL_SIZE) + 1)
    n_bin_y = int(np.ceil((max_y - min_y) / CELL_SIZE) + 1)
    grid = np.zeros((n_bin_x, n_bin_y))

    for group in groups:
        trajectory = group["center_of_mass"]

        x = trajectory[:, 1]
        y = trajectory[:, 2]

        # plt.plot(x, y)
        # plt.axis("equal")
        # plt.show()

        nx = np.ceil((x - min_x) / CELL_SIZE).astype("int")
        ny = np.ceil((y - min_y) / CELL_SIZE).astype("int")

        in_limit = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)

        nx = nx[in_limit]
        ny = ny[in_limit]

        grid[nx, ny] += 1

    max_val = np.max(grid)
    grid /= max_val

    xi = np.linspace(min_x, max_x, n_bin_x)
    yi = np.linspace(min_y, max_y, n_bin_y)

    fig, ax = plt.subplots(figsize=(12, 6))
    cmesh = ax.pcolormesh(xi, yi, grid.T, cmap="viridis")

    # plot rectangle for sources
    for source_name, source in SOURCES_ATC.items():
        rect = Rectangle(
            (source["xmin"], source["ymin"]),
            source["xmax"] - source["xmin"],
            source["ymax"] - source["ymin"],
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

        ax.text(
            source["xmin"] + 200,
            source["ymin"] + 200,
            source_name,
            fontsize=12,
            color="red",
        )

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    # plt.show()
    plt.savefig(f"../data/figures/03_01_sources_sinks_atc.png", dpi=300)
    plt.close()
