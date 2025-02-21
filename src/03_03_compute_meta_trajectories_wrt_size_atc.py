import matplotlib.pyplot as plt
import numpy as np

from utils import (
    pickle_load,
    pickle_save,
    filter_groups_by_size,
    compute_time_bin_average_trajectory,
)
from parameters import SOURCES_ATC


if __name__ == "__main__":
    source_to_sink_groups = pickle_load(
        "../data/intermediate/03_02_source_to_sink_trajectories_atc.pkl"
    )

    N_POINTS_TRAJECTORY = 100
    MIN_N_TRAJECTORIES = 2

    CELL_SIZE = 500  # mm
    MAX_N_TRAJECTORIES_BAD = 1
    RATIO_BAD_CELLS = 0.1

    meta_trajectories = {}

    for (source, sink), groups in source_to_sink_groups.items():

        meta_trajectories[(source, sink)] = {}

        for size in [1, 2, 3, 4, 5, 6, 7]:

            size_groups = filter_groups_by_size(groups, size)

            source_to_sink_trajectories = []

            for group in size_groups:

                for member in group["members"]:

                    trajectory = member["trajectory_source_to_sink"]

                    x = trajectory[:, 1]
                    y = trajectory[:, 2]

                    average_trajectory = compute_time_bin_average_trajectory(
                        trajectory, N_POINTS_TRAJECTORY
                    )

                    source_to_sink_trajectories.append(average_trajectory)

            source_to_sink_trajectories = np.array(source_to_sink_trajectories)

            if len(source_to_sink_trajectories) < MIN_N_TRAJECTORIES:
                continue

            min_x = np.nanmin(source_to_sink_trajectories[:, :, 1])
            max_x = np.nanmax(source_to_sink_trajectories[:, :, 1])
            min_y = np.nanmin(source_to_sink_trajectories[:, :, 2])
            max_y = np.nanmax(source_to_sink_trajectories[:, :, 2])

            n_trajectories = len(source_to_sink_trajectories)

            # remove the "bad" trajectories
            n_bin_x = int(np.ceil((max_x - min_x) / CELL_SIZE) + 1)
            n_bin_y = int(np.ceil((max_y - min_y) / CELL_SIZE) + 1)
            grid = np.zeros((n_bin_x, n_bin_y, n_trajectories))

            for i, trajectory in enumerate(source_to_sink_trajectories):
                x = trajectory[:, 1]
                y = trajectory[:, 2]

                x = x[~np.isnan(x)]
                y = y[~np.isnan(y)]

                nx = (x - min_x) / CELL_SIZE
                ny = (y - min_y) / CELL_SIZE

                nx = np.ceil(nx).astype("int")
                ny = np.ceil(ny).astype("int")

                in_limit = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)

                nx = nx[in_limit]
                ny = ny[in_limit]

                grid[nx, ny, i] = 1

            # find grid cells with few trajectories
            grid_n_trajectories = np.sum(grid, axis=2)
            mask_grid_cells_few_trajectories = (grid_n_trajectories > 0) & (
                grid_n_trajectories <= MAX_N_TRAJECTORIES_BAD
            )

            good_trajectories = []
            for i, trajectory in enumerate(source_to_sink_trajectories):
                count_bad_cells = np.sum(grid[mask_grid_cells_few_trajectories, i])
                total_cells = np.sum(grid[:, :, i])
                ratio_bad = count_bad_cells / total_cells

                if ratio_bad < RATIO_BAD_CELLS:
                    good_trajectories.append(trajectory)

            good_trajectories = np.array(good_trajectories)

            # good_trajectories = source_to_sink_trajectories

            if len(good_trajectories) < MIN_N_TRAJECTORIES:
                continue

            min_x = np.nanmin(good_trajectories[:, :, 1])
            max_x = np.nanmax(good_trajectories[:, :, 1])
            min_y = np.nanmin(good_trajectories[:, :, 2])
            max_y = np.nanmax(good_trajectories[:, :, 2])

            meta_trajectory = np.nanmean(good_trajectories, axis=0)

            meta_trajectories[(source, sink)][size] = meta_trajectory

        # Trajectories
        fig, ax = plt.subplots(figsize=(12, 6))

        for size, trajectory in meta_trajectories[(source, sink)].items():
            ax.plot(
                trajectory[:, 1],
                trajectory[:, 2],
                linewidth=2,
                label=size,
            )

        for s, c in zip([source, sink], ["blue", "orange"]):
            source_data = SOURCES_ATC[s]
            ax.add_patch(
                plt.Rectangle(
                    (source_data["xmin"], source_data["ymin"]),
                    source_data["xmax"] - source_data["xmin"],
                    source_data["ymax"] - source_data["ymin"],
                    edgecolor="black",
                    facecolor=c,
                    alpha=0.5,
                    zorder=10,
                )
            )

        ax.legend()
        ax.set_aspect("equal")

        plt.tight_layout()
        plt.savefig(
            f"../data/figures/03_03_meta_trajectories_wrt_size_atc/03_04_{source}_{sink}_meta_trajectory.png"
        )
        plt.close()

        # # Velocities
        # fig, ax = plt.subplots(figsize=(12, 6))

        # for size, trajectory in meta_trajectories[(source, sink)].items():
        #     vx = trajectory[:, 5]
        #     vy = trajectory[:, 6]

        #     v_mag = np.sqrt(vx**2 + vy**2)

        #     ax.plot(
        #         trajectory[:, 0],
        #         v_mag,
        #         linewidth=2,
        #         label=size,
        #     )

        # ax.legend()
        # plt.tight_layout()
        # plt.savefig(
        #     f"../data/figures/02_05_meta_trajectories_velocities_wrt_size/{day}/02_04_{source}_{sink}_velocities.png"
        # )
        # plt.close()
