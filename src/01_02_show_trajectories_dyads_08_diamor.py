import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from utils import pickle_load, filter_groups_by_day, filter_groups_by_social_binding
from parameters import SOURCES_DIAMOR


if __name__ == "__main__":

    groups = pickle_load("../data/intermediate/01_01_groups_diamor.pkl")

    groups = filter_groups_by_day(groups, "08")

    fig, ax = plt.subplots()
    for interaction, c in zip([0, 1, 2, 3], ["red", "blue", "green", "orange"]):
        interaction_groups = filter_groups_by_social_binding(groups, interaction)
        print(len(interaction_groups))

        for group in interaction_groups:
            trajectory = group["center_of_mass"]

            ax.plot(
                trajectory[:, 1],
                trajectory[:, 2],
                color=c,
                label=f"Interaction {interaction}",
            )

    for source in SOURCES_DIAMOR["08"].values():
        ax.add_patch(
            plt.Rectangle(
                (source["xmin"], source["ymin"]),
                source["xmax"] - source["xmin"],
                source["ymax"] - source["ymin"],
                edgecolor="red",
                facecolor="red",
                alpha=0.5,
            )
        )

    ax.set_aspect("equal")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.show()
