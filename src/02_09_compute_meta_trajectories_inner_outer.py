import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import scienceplots

plt.style.use("science")


from utils import (
    pickle_load,
    pickle_save,
    filter_groups_by_size,
    compute_space_bin_average_trajectory,
    transform_trajectories_to_reference_frame,
    compute_center_of_mass_trajectory,
    compute_distance,
    filter_bad_trajectories,
    compute_average_interpersonal_distance,
)

from parameters import (
    SOURCES_DIAMOR,
    MIN_N_TRAJECTORIES,
    SIZE_BIN,
    CELL_SIZE,
    MIN_TRAJECTORIES_BAD,
    RATIO_BAD_CELLS,
    TURN_TYPE_DIAMOR,
)


def find_inner_outer(trajectory_1, trajectory_2, turn_type):
    """
    Find inner and outer trajectory.

    Parameters
    ----------
    trajectory_1 : np.array
        Trajectory 1.
    trajectory_2 : np.array
        Trajectory 2.

    Returns
    -------
    inner_trajectory : np.array
        Inner trajectory.
    outer_trajectory : np.array
        Outer trajectory.
    """
    trajectory_com = compute_center_of_mass_trajectory([trajectory_1, trajectory_2])
    _, [trajectory_1_aligned, trajectory_2_aligned] = (
        transform_trajectories_to_reference_frame(
            trajectory_com,
            [trajectory_1, trajectory_2],
            nullify_velocities=False,
            axis="y",
        )
    )

    # find left and right trajectory
    x_1 = trajectory_1_aligned[:, 1]
    x_2 = trajectory_2_aligned[:, 1]
    d_x = x_2 - x_1
    if np.mean(d_x) < 0:
        trajectory_left = trajectory_2
        trajectory_right = trajectory_1
        trajectory_left_aligned = trajectory_2_aligned
        trajectory_right_aligned = trajectory_1_aligned
    else:
        trajectory_left = trajectory_1
        trajectory_right = trajectory_2
        trajectory_left_aligned = trajectory_1_aligned
        trajectory_right_aligned = trajectory_2_aligned

    trajectory_inner = trajectory_right
    trajectory_outer = trajectory_left
    trajectory_inner_aligned = trajectory_right_aligned
    trajectory_outer_aligned = trajectory_left_aligned

    if turn_type == "left":
        trajectory_inner = trajectory_left
        trajectory_outer = trajectory_right
        trajectory_inner_aligned = trajectory_left_aligned
        trajectory_outer_aligned = trajectory_right_aligned

    # fig, axes = plt.subplots(1, 2)

    # axes[0].scatter(
    #     trajectory_inner[:, 1],
    #     trajectory_inner[:, 2],
    #     color="blue",
    #     label="inner",
    #     s=5,
    # )
    # axes[0].scatter(
    #     trajectory_outer[:, 1],
    #     trajectory_outer[:, 2],
    #     color="orange",
    #     label="outer",
    #     s=5,
    # )
    # axes[0].scatter(trajectory_com[:, 1], trajectory_com[:, 2], color="black", s=5)

    # axes[1].plot(
    #     trajectory_inner_aligned[:, 1], trajectory_inner_aligned[:, 2], color="blue"
    # )
    # axes[1].plot(
    #     trajectory_outer_aligned[:, 1], trajectory_outer_aligned[:, 2], color="orange"
    # )

    # axes[0].set_aspect("equal")
    # axes[1].set_aspect("equal")
    # axes[0].legend()
    # plt.show()

    return trajectory_inner, trajectory_outer


if __name__ == "__main__":
    source_to_sink_groups = pickle_load(
        "../data/intermediate/02_02_source_to_sink_groups_diamor.pkl"
    )

    meta_trajectories = {}
    distances = {}

    for day in ["06", "08"]:

        meta_trajectories[day] = {}
        distances[day] = {}

        for (source, sink), groups in tqdm(source_to_sink_groups[day].items()):

            turn_type = TURN_TYPE_DIAMOR[day][(source, sink)]

            meta_trajectories[day][(source, sink)] = {}

            # find distance between source and sink
            source_data = SOURCES_DIAMOR[day][source]
            sink_data = SOURCES_DIAMOR[day][sink]

            distance = compute_distance(source_data, sink_data)

            n_points_trajectory = int(np.floor(distance / SIZE_BIN))

            dyads = filter_groups_by_size(groups, 2)

            inner_trajectories = []
            outer_trajectories = []
            com_trajectories = []
            all_trajectories = []
            interpersonal_distances = []

            for dyad in dyads:

                trajectory_innner, trajectory_outer = find_inner_outer(
                    dyad["members"][0]["trajectory_source_to_sink"],
                    dyad["members"][1]["trajectory_source_to_sink"],
                    turn_type,
                )

                average_trajectory_inner = compute_space_bin_average_trajectory(
                    trajectory_innner, n_points_trajectory, time="average"
                )
                average_trajectory_outer = compute_space_bin_average_trajectory(
                    trajectory_outer, n_points_trajectory, time="average"
                )

                average_trajectory_com = compute_space_bin_average_trajectory(
                    compute_center_of_mass_trajectory(
                        [trajectory_innner, trajectory_outer]
                    ),
                    n_points_trajectory,
                    time="average",
                )

                inner_trajectories.append(average_trajectory_inner)
                outer_trajectories.append(average_trajectory_outer)
                com_trajectories.append(average_trajectory_com)

                d = compute_average_interpersonal_distance(
                    trajectory_innner, trajectory_outer
                )
                interpersonal_distances.append(d)

            if len(interpersonal_distances) > 0:
                distances[day][(source, sink)] = np.nanmean(interpersonal_distances)

            inner_trajectories = np.array(inner_trajectories)
            outer_trajectories = np.array(outer_trajectories)
            com_trajectories = np.array(com_trajectories)

            if len(inner_trajectories) < MIN_N_TRAJECTORIES:
                continue
            if len(outer_trajectories) < MIN_N_TRAJECTORIES:
                continue
            if len(com_trajectories) < MIN_N_TRAJECTORIES:
                continue

            good_inner_trajectories = inner_trajectories
            good_outer_trajectories = outer_trajectories
            good_com_trajectories = com_trajectories

            # good_inner_trajectories = filter_bad_trajectories(
            #     inner_trajectories, CELL_SIZE, MIN_TRAJECTORIES_BAD, RATIO_BAD_CELLS
            # )
            # good_outer_trajectories = filter_bad_trajectories(
            #     outer_trajectories, CELL_SIZE, MIN_TRAJECTORIES_BAD, RATIO_BAD_CELLS
            # )
            # good_com_trajectories = filter_bad_trajectories(
            #     com_trajectories, CELL_SIZE, MIN_TRAJECTORIES_BAD, RATIO_BAD_CELLS
            # )

            # if len(good_inner_trajectories) < MIN_N_TRAJECTORIES:
            #     continue
            # if len(good_outer_trajectories) < MIN_N_TRAJECTORIES:
            #     continue
            # if len(good_com_trajectories) < MIN_N_TRAJECTORIES:
            #     continue

            all_trajectories.extend(inner_trajectories)
            all_trajectories.extend(outer_trajectories)
            all_trajectories = np.array(all_trajectories)

            meta_trajectory_inner = np.nanmean(good_inner_trajectories, axis=0)
            meta_trajectory_outer = np.nanmean(good_outer_trajectories, axis=0)
            meta_trajectory_all = np.nanmean(all_trajectories, axis=0)
            meta_trajectory_com = np.nanmean(good_com_trajectories, axis=0)

            meta_trajectories[day][(source, sink)]["inner"] = meta_trajectory_inner
            meta_trajectories[day][(source, sink)]["outer"] = meta_trajectory_outer
            meta_trajectories[day][(source, sink)]["all"] = meta_trajectory_all
            meta_trajectories[day][(source, sink)]["com"] = meta_trajectory_com

            fig, ax = plt.subplots(figsize=(12, 6))

            for trajectory in good_inner_trajectories:
                ax.plot(
                    trajectory[:, 1],
                    trajectory[:, 2],
                    color="blue",
                    linewidth=0.5,
                    alpha=0.3,
                )

            for trajectory in good_outer_trajectories:
                ax.plot(
                    trajectory[:, 1],
                    trajectory[:, 2],
                    color="orange",
                    linewidth=0.5,
                    alpha=0.3,
                )

            ax.plot(
                meta_trajectory_inner[:, 1],
                meta_trajectory_inner[:, 2],
                color="blue",
                label="inner",
                linewidth=2,
            )
            ax.plot(
                meta_trajectory_outer[:, 1],
                meta_trajectory_outer[:, 2],
                color="orange",
                label="outer",
                linewidth=2,
            )

            ax.set_aspect("equal")
            ax.legend()
            plt.savefig(
                f"../data/figures/02_09_meta_trajectories_inner_outer/{day}_{source}_{sink}.pdf"
            )
            plt.close()

    pickle_save(
        meta_trajectories,
        "../data/intermediate/02_09_meta_trajectories_inner_outer.pkl",
    )

    pickle_save(
        distances,
        "../data/intermediate/02_09_distances_inner_outer.pkl",
    )
