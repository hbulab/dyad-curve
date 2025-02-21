import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from utils import (
    pickle_load,
    pickle_save,
    filter_groups_by_day,
    compute_mask_in_source,
    compute_trajectory_source_to_sink,
)
from parameters import SOURCES_ATC, BOUNDARIES_ATC


if __name__ == "__main__":

    groups = pickle_load("../data/intermediate/01_01_groups_atc.pkl")

    source_to_sink_groups = {}

    for group in groups:
        trajectory = group["center_of_mass"]

        # find if the trajectory goes from one source to another
        in_sources = []
        for source_name, source in SOURCES_ATC.items():
            in_source_mask = compute_mask_in_source(trajectory, source)
            if np.any(in_source_mask):
                in_sources += [(source_name, source)]

        if len(in_sources) != 2:
            continue

        in_source_1 = compute_mask_in_source(trajectory, in_sources[0][1])
        in_source_2 = compute_mask_in_source(trajectory, in_sources[1][1])

        t_min_source_1 = np.min(trajectory[in_source_1, 0])
        t_min_source_2 = np.min(trajectory[in_source_2, 0])

        source, sink = (
            (in_sources[0], in_sources[1])
            if t_min_source_1 < t_min_source_2
            else (in_sources[1], in_sources[0])
        )

        source_to_sink = (source[0], sink[0])

        trajectory_source_to_sink = compute_trajectory_source_to_sink(
            trajectory, source[1], sink[1]
        )

        member_not_ok = False
        for members in group["members"]:
            members["trajectory_source_to_sink"] = compute_trajectory_source_to_sink(
                members["trajectory"], source[1], sink[1]
            )
            if members["trajectory_source_to_sink"] is None:
                member_not_ok = True
                break

        if member_not_ok:
            continue

        group["center_of_mass_source_to_sink"] = trajectory_source_to_sink

        if source_to_sink not in source_to_sink_groups:
            source_to_sink_groups[source_to_sink] = []

        source_to_sink_groups[source_to_sink] += [group]

        # fig, ax = plt.subplots()
        # ax.plot(
        #     trajectory_source_to_sink[:, 1],
        #     trajectory_source_to_sink[:, 2],
        #     color="black",
        # )

        # for s in source_to_sink:
        #     source_data = SOURCES_ATC[s]
        #     ax.add_patch(
        #         plt.Rectangle(
        #             (source_data["xmin"], source_data["ymin"]),
        #             source_data["xmax"] - source_data["xmin"],
        #             source_data["ymax"] - source_data["ymin"],
        #             edgecolor="red",
        #             facecolor="red",
        #             alpha=0.5,
        #         )
        #     )

        # ax.set_xlim(BOUNDARIES_ATC["xmin"], BOUNDARIES_ATC["xmax"])
        # ax.set_ylim(BOUNDARIES_ATC["ymin"], BOUNDARIES_ATC["ymax"])
        # ax.set_aspect("equal")
        # plt.show()

    # save the turning groups
    pickle_save(
        source_to_sink_groups,
        "../data/intermediate/03_02_source_to_sink_trajectories_atc.pkl",
    )
