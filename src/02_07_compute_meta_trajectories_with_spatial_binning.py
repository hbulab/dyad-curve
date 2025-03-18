from matplotlib.patches import Rectangle
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
    compute_distance,
    filter_bad_trajectories,
)
from parameters import (
    SOURCES_DIAMOR,
    MIN_N_TRAJECTORIES,
    SIZE_BIN,
    CELL_SIZE,
    MIN_TRAJECTORIES_BAD,
    RATIO_BAD_CELLS,
)


if __name__ == "__main__":
    source_to_sink_groups = pickle_load(
        "../data/intermediate/02_02_source_to_sink_groups_diamor.pkl"
    )

    meta_trajectories = {}
    filtered_trajectories = {}

    for day in ["06", "08"]:

        meta_trajectories[day] = {}
        filtered_trajectories[day] = {}

        for (source, sink), groups in tqdm(source_to_sink_groups[day].items()):

            # for source, sink in tqdm(STRONG_TURN_PAIRS_DIAMOR[day]):

            groups = source_to_sink_groups[day][(source, sink)]

            meta_trajectories[day][(source, sink)] = {}

            # # only turning groups
            # if (source, sink) in STRAIGHT_PAIRS_DIAMOR[day]:
            #     continue

            # find distance between source and sink
            source_data = SOURCES_DIAMOR[day][source]
            sink_data = SOURCES_DIAMOR[day][sink]

            distance = compute_distance(source_data, sink_data)

            n_points_trajectory = int(np.floor(distance / SIZE_BIN))

            all_trajectories = []

            fig, ax = plt.subplots(figsize=(10, 10))

            for size in [1, 2]:

                size_groups = filter_groups_by_size(groups, size)

                source_to_sink_trajectories = []
                ids = []

                for group in size_groups:

                    for member in group["members"]:

                        trajectory = member["trajectory_source_to_sink"]

                        average_trajectory = compute_space_bin_average_trajectory(
                            trajectory, n_points_trajectory, time="average"
                        )
                        source_to_sink_trajectories.append(average_trajectory)
                        ids.append((group["id"], member["id"]))

                source_to_sink_trajectories = np.array(source_to_sink_trajectories)

                if len(source_to_sink_trajectories) < MIN_N_TRAJECTORIES:
                    continue

                # good_trajectories = source_to_sink_trajectories
                good_trajectories, good_ids = filter_bad_trajectories(
                    source_to_sink_trajectories,
                    ids,
                    CELL_SIZE,
                    MIN_TRAJECTORIES_BAD,
                    RATIO_BAD_CELLS,
                )

                if len(good_trajectories) < MIN_N_TRAJECTORIES:
                    continue

                # keep track of filtered trajectories for dyads
                if size == 2:
                    filtered_trajectories[day][(source, sink)] = good_ids

                all_trajectories.extend(good_trajectories)

                meta_trajectory = np.nanmean(good_trajectories, axis=0)
                meta_trajectories[day][(source, sink)][size] = meta_trajectory

                for trajectory in good_trajectories:
                    ax.plot(
                        trajectory[:, 1] / 1000,
                        trajectory[:, 2] / 1000,
                        linewidth=0.5,
                        alpha=0.1,
                        color="orange" if size == 1 else "blue",
                    )

                ax.plot(
                    meta_trajectory[:, 1] / 1000,
                    meta_trajectory[:, 2] / 1000,
                    linewidth=2,
                    color="orange" if size == 1 else "blue",
                    label=f"$T_{{{source},{sink}}}^{{{'ind' if size == 1 else 'dyad'}}}$",
                    zorder=10,
                )

                # draw sources
                for s, c in zip([source, sink], ["green", "purple"]):
                    source_data = SOURCES_DIAMOR[day][s]
                    rect_outer = Rectangle(
                        (source_data["xmin"] / 1000, source_data["ymin"] / 1000),
                        source_data["xmax"] / 1000 - source_data["xmin"] / 1000,
                        source_data["ymax"] / 1000 - source_data["ymin"] / 1000,
                        edgecolor=c,
                        facecolor="none",
                    )
                    rect_inner = Rectangle(
                        (source_data["xmin"] / 1000, source_data["ymin"] / 1000),
                        source_data["xmax"] / 1000 - source_data["xmin"] / 1000,
                        source_data["ymax"] / 1000 - source_data["ymin"] / 1000,
                        edgecolor="none",
                        facecolor=c,
                        alpha=0.5,
                    )
                    ax.add_patch(rect_outer)
                    ax.add_patch(rect_inner)
                    ax.text(
                        source_data["xmin"] / 1000 + 0.2,
                        source_data["ymin"] / 1000 + 0.2,
                        s,
                        fontsize=12,
                        color="black",
                    )
            if len(all_trajectories) == 0:
                continue

            ax.legend()
            ax.set_aspect("equal")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")

            plt.tight_layout()
            plt.savefig(
                f"../data/figures/02_07_meta_trajectories/{day}_{source}_{sink}.pdf"
            )
            plt.close()

            all_trajectories = np.array(all_trajectories)
            if len(all_trajectories) < MIN_N_TRAJECTORIES:
                continue
            meta_trajectory = np.nanmean(all_trajectories, axis=0)
            meta_trajectories[day][(source, sink)]["all"] = meta_trajectory

    pickle_save(
        meta_trajectories, "../data/intermediate/02_07_meta_trajectories_diamor.pkl"
    )

    pickle_save(
        filtered_trajectories,
        "../data/intermediate/02_07_filtered_trajectories_diamor.pkl",
    )
