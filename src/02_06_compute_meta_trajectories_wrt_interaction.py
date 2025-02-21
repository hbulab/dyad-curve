import matplotlib.pyplot as plt
import numpy as np

from utils import (
    pickle_load,
    pickle_save,
    filter_groups_by_social_binding,
    compute_time_bin_average_trajectory,
)
from parameters import STRAIGHT_PAIRS_DIAMOR, SOURCES_DIAMOR


if __name__ == "__main__":
    source_to_sink_groups = pickle_load(
        "../data/intermediate/02_02_source_to_sink_groups_diamor.pkl"
    )

    N_POINTS_TRAJECTORY = 100
    MIN_N_TRAJECTORIES = 2

    CELL_SIZE = 500  # mm
    MAX_N_TRAJECTORIES_BAD = 1
    RATIO_BAD_CELLS = 0.1

    meta_trajectories = {}

    for day in ["06", "08"]:

        meta_trajectories[day] = {}

        for (source, sink), groups in source_to_sink_groups[day].items():

            meta_trajectories[day][(source, sink)] = {}

            # only turning groups
            if (source, sink) in STRAIGHT_PAIRS_DIAMOR[day]:
                continue

            for interaction in [0, 1, 2, 3]:

                interaction_groups = filter_groups_by_social_binding(
                    groups, interaction
                )

                source_to_sink_trajectories = []

                for group in interaction_groups:

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

                # good_trajectories = []
                # for i, trajectory in enumerate(source_to_sink_trajectories):
                #     count_bad_cells = np.sum(grid[mask_grid_cells_few_trajectories, i])
                #     total_cells = np.sum(grid[:, :, i])
                #     ratio_bad = count_bad_cells / total_cells

                #     if ratio_bad < RATIO_BAD_CELLS:
                #         good_trajectories.append(trajectory)

                # good_trajectories = np.array(good_trajectories)

                good_trajectories = source_to_sink_trajectories

                if len(good_trajectories) < MIN_N_TRAJECTORIES:
                    continue

                min_x = np.nanmin(good_trajectories[:, :, 1])
                max_x = np.nanmax(good_trajectories[:, :, 1])
                min_y = np.nanmin(good_trajectories[:, :, 2])
                max_y = np.nanmax(good_trajectories[:, :, 2])

                meta_trajectory = np.nanmean(good_trajectories, axis=0)

                meta_trajectories[day][(source, sink)][interaction] = meta_trajectory

            # Trajectories
            fig, ax = plt.subplots(figsize=(12, 6))

            for interaction, trajectory in meta_trajectories[day][
                (source, sink)
            ].items():
                ax.plot(
                    trajectory[:, 1],
                    trajectory[:, 2],
                    linewidth=2,
                    label=interaction,
                )

            for s, c in zip([source, sink], ["blue", "orange"]):
                source_data = SOURCES_DIAMOR[day][s]
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
                f"../data/figures/02_06_meta_trajectories_wrt_interaction/{day}/02_06_{source}_{sink}_meta_trajectory.png"
            )
            plt.close()
