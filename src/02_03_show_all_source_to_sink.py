import matplotlib.pyplot as plt
import numpy as np

from utils import pickle_load
from parameters import SOURCES_DIAMOR, BOUNDARIES_DIAMOR, STRAIGHT_PAIRS_DIAMOR


if __name__ == "__main__":

    source_to_sink_groups = pickle_load(
        "../data/intermediate/02_02_source_to_sink_groups_diamor.pkl"
    )

    for day in ["06", "08"]:
        for (source, sink), groups in source_to_sink_groups[day].items():

            fig, ax = plt.subplots()

            for group in groups:
                trajectory = group["center_of_mass_source_to_sink"]
                ax.plot(trajectory[:, 1], trajectory[:, 2], color="black")

            for s in [source, sink]:
                source_data = SOURCES_DIAMOR[day][s]
                ax.add_patch(
                    plt.Rectangle(
                        (source_data["xmin"], source_data["ymin"]),
                        source_data["xmax"] - source_data["xmin"],
                        source_data["ymax"] - source_data["ymin"],
                        edgecolor="red",
                        facecolor="red",
                        alpha=0.5,
                    )
                )

            ax.set_xlim(BOUNDARIES_DIAMOR[day]["xmin"], BOUNDARIES_DIAMOR[day]["xmax"])
            ax.set_ylim(BOUNDARIES_DIAMOR[day]["ymin"], BOUNDARIES_DIAMOR[day]["ymax"])

            ax.set_aspect("equal")

            ax.set_title(f"Source: {source} - Sink: {sink}")

            plt.show()
