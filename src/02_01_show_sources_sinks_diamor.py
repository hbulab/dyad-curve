import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from utils import pickle_load, filter_groups_by_day
from parameters import SOURCES_DIAMOR

import scienceplots

plt.style.use("science")


if __name__ == "__main__":

    groups = pickle_load("../data/intermediate/01_01_groups_diamor.pkl")

    for day in ["06", "08"]:
        day_groups = filter_groups_by_day(groups, day)
        # day_groups = filter_groups_by_size(day_groups, 2)

        # find the env boundaries
        mins_x, mins_y, maxs_x, maxs_y = [], [], [], []

        for group in day_groups:
            for member in group["members"]:
                trajectory = member["trajectory"]
                if len(trajectory) <= 0:
                    continue
                mins_x.append(np.min(trajectory[:, 1] / 1000))
                maxs_x.append(np.max(trajectory[:, 1] / 1000))
                mins_y.append(np.min(trajectory[:, 2] / 1000))
                maxs_y.append(np.max(trajectory[:, 2] / 1000))

        min_x = np.min(mins_x)
        max_x = np.max(maxs_x)
        min_y = np.min(mins_y)
        max_y = np.max(maxs_y)

        CELL_SIZE = 0.1

        n_bin_x = int(np.ceil((max_x - min_x) / CELL_SIZE) + 1)
        n_bin_y = int(np.ceil((max_y - min_y) / CELL_SIZE) + 1)
        grid = np.zeros((n_bin_x, n_bin_y))

        for group in day_groups:
            trajectory = group["center_of_mass"]

            x = trajectory[:, 1] / 1000
            y = trajectory[:, 2] / 1000

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
        cmesh = ax.pcolormesh(xi, yi, grid.T, cmap="inferno_r")

        # plot rectangle for sources
        for source_name, source in SOURCES_DIAMOR[day].items():
            rect_outer = Rectangle(
                (source["xmin"] / 1000, source["ymin"] / 1000),
                source["xmax"] / 1000 - source["xmin"] / 1000,
                source["ymax"] / 1000 - source["ymin"] / 1000,
                edgecolor="green",
                facecolor="none",
            )
            rect_inner = Rectangle(
                (source["xmin"] / 1000, source["ymin"] / 1000),
                source["xmax"] / 1000 - source["xmin"] / 1000,
                source["ymax"] / 1000 - source["ymin"] / 1000,
                edgecolor="none",
                facecolor="green",
                alpha=0.5,
            )
            ax.add_patch(rect_outer)
            ax.add_patch(rect_inner)

            ax.text(
                source["xmin"] / 1000 + 0.2,
                source["ymin"] / 1000 + 0.2,
                source_name,
                fontsize=12,
                color="black",
            )

        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        # plt.show()
        plt.savefig(f"../data/figures/02_01_sources_sinks_diamor_{day}.png", dpi=300)
        plt.close()
