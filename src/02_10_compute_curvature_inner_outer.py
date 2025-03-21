import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.stats import linregress
import seaborn as sns
import pandas as pd

import scienceplots

plt.style.use("science")


from utils import (
    pickle_load,
    compute_binned_values,
    compute_curvature,
    compute_angular_velocity,
    fit_piecewise_linear,
    plot_piecewise_linear,
)

from parameters import (
    WINDOW_SIZE,
    TURN_TYPE_DIAMOR,
    N_BINS,
    TURNING_X_MAX,
    TURNING_X_MIN,
)


if __name__ == "__main__":

    meta_trajectories = pickle_load(
        "../data/intermediate/02_07_meta_trajectories_diamor.pkl"
    )
    meta_trajectories_inner_outer = pickle_load(
        "../data/intermediate/02_09_meta_trajectories_inner_outer.pkl"
    )
    interpersonal_distances = pickle_load(
        "../data/intermediate/02_09_distances_inner_outer.pkl"
    )

    df_values = {
        "velocity": [],
        "curvature": [],
        "angular_velocity": [],
        "angular_velocity_from_curvature": [],
        "distance": [],
        "time_step": [],
        "x": [],
        "y": [],
        "id": [],
        "type": [],
        "turning": [],
        "straight": [],
    }

    for day in ["06", "08"]:
        for source, sink in tqdm(meta_trajectories_inner_outer[day].keys()):

            if TURN_TYPE_DIAMOR[day][(source, sink)] == "straight":
                continue

            fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

            for pos, color in zip(["com", "inner", "outer"], ["black", "blue", "red"]):

                if pos not in meta_trajectories_inner_outer[day][(source, sink)]:
                    continue

                meta_trajectory_position = meta_trajectories_inner_outer[day][
                    (source, sink)
                ][pos]
                if len(meta_trajectory_position) < WINDOW_SIZE:
                    continue
                velocity, curvature = compute_curvature(
                    meta_trajectory_position, WINDOW_SIZE
                )

                t = np.linspace(0, 1, len(velocity))

                df_values["id"].extend([f"{day}_{source}_{sink}"] * len(velocity))
                df_values["type"].extend([pos] * len(velocity))
                df_values["time_step"].extend(np.arange(len(velocity)))
                df_values["distance"].extend(
                    [interpersonal_distances[day][(source, sink)]] * len(velocity)
                )

                # portion of trajectory in straight part
                straight_portion = (meta_trajectory_position[:, 1] > TURNING_X_MIN) & (
                    meta_trajectory_position[:, 1] < TURNING_X_MAX
                )
                # turning portion
                turning_portion = meta_trajectory_position[:, 1] < TURNING_X_MIN

                df_values["turning"].extend(turning_portion.tolist())
                df_values["straight"].extend(straight_portion.tolist())

                curvature = np.abs(curvature) * 1000
                velocity = velocity / 1000

                angular_velocity = np.abs(
                    compute_angular_velocity(meta_trajectory_position, WINDOW_SIZE)
                )

                df_values["velocity"].extend(velocity.tolist())
                df_values["curvature"].extend(curvature.tolist())
                df_values["angular_velocity"].extend(angular_velocity.tolist())
                df_values["angular_velocity_from_curvature"].extend(
                    (velocity / (1 / curvature)).tolist()
                )

                x = meta_trajectory_position[:, 1] / 1000
                y = meta_trajectory_position[:, 2] / 1000

                df_values["x"].extend(x.tolist())
                df_values["y"].extend(y.tolist())

                axes[0].plot(t, velocity, color=color, label=pos)
                axes[1].plot(t, curvature, color=color, label=pos)
                axes[2].plot(t, angular_velocity, color=color, label=pos)

            axes[0].set_ylabel("Velocity [m/s]")
            axes[1].set_ylabel("Curvature [1/m]")
            axes[2].set_ylabel("Angular velocity [rad/s]")
            axes[2].set_xlabel("Time [s]")

            for i in range(len(axes)):
                axes[i].legend()
                axes[i].grid(color="gray", linestyle="--", linewidth=0.5)

            plt.savefig(
                f"../data/figures/02_10_velocity_curvature_inner_outer/{day}_{source}_{sink}.pdf"
            )
            plt.close()

            # plot trajectory with curvature and trajectory with velocity
            sc_curvature = None
            sc_velocity = None

            fig, ax = plt.subplots(3, 1, figsize=(12, 6), sharex=True, sharey=True)

            for pos, m in zip(["inner", "outer"], ["o", "s"]):

                if pos not in meta_trajectories_inner_outer[day][(source, sink)]:
                    continue

                meta_trajectory_position = meta_trajectories_inner_outer[day][
                    (source, sink)
                ][pos]
                if len(meta_trajectory_position) < WINDOW_SIZE:
                    continue
                velocity, curvature = compute_curvature(
                    meta_trajectory_position, WINDOW_SIZE
                )
                angular_velocity = np.abs(
                    compute_angular_velocity(meta_trajectory_position, WINDOW_SIZE)
                )

                curvature = np.abs(curvature) * 1000
                velocity = velocity / 1000

                sc_curvature = ax[0].scatter(
                    meta_trajectory_position[:, 1],
                    meta_trajectory_position[:, 2],
                    label=f"{pos}",
                    c=curvature,
                    marker=m,
                    s=10,
                    cmap="viridis",
                    vmin=0,
                    vmax=0.2,
                    alpha=0.6,
                )

                sc_velocity = ax[1].scatter(
                    meta_trajectory_position[:, 1],
                    meta_trajectory_position[:, 2],
                    label=f"{pos}",
                    c=velocity,
                    marker=m,
                    s=10,
                    cmap="viridis",
                    vmin=1,
                    vmax=1.5,
                    alpha=0.6,
                )

                sc_angular_velocity = ax[2].scatter(
                    meta_trajectory_position[:, 1],
                    meta_trajectory_position[:, 2],
                    label=f"{pos}",
                    c=np.log(angular_velocity),
                    marker=m,
                    s=10,
                    cmap="viridis",
                    vmin=np.log(0.01),
                    vmax=np.log(0.5),
                    alpha=0.6,
                )

            ax[0].set_title(f"Day {day}, {source} - {sink}")
            ax[0].set_xlabel("X [mm]")
            ax[0].set_ylabel("Y [mm]")
            ax[0].legend()
            ax[0].set_aspect("equal")

            ax[1].set_title(f"Day {day}, {source} - {sink}")
            ax[1].set_xlabel("X [mm]")
            ax[1].set_ylabel("Y [mm]")
            ax[1].legend()
            ax[1].set_aspect("equal")

            ax[2].set_title(f"Day {day}, {source} - {sink}")
            ax[2].set_xlabel("X [mm]")
            ax[2].set_ylabel("Y [mm]")
            ax[2].legend()
            ax[2].set_aspect("equal")

            # Add colorbars
            fig.colorbar(sc_curvature, ax=ax[0], label="Curvature [1/m]")
            fig.colorbar(sc_velocity, ax=ax[1], label="Velocity [m/s]")
            fig.colorbar(
                sc_angular_velocity, ax=ax[2], label="Log angular velocity [rad/s]"
            )

            plt.tight_layout()
            plt.savefig(
                f"../data/figures/02_10_trajectories_with_velocity_curvature_inner_outer/{day}_{source}_{sink}_trajectory.pdf"
            )
            plt.close()

            # compare inner and outer angular velocity

            fig, ax = plt.subplots(2, 1, figsize=(8, 8))

            angular_velocity_inner = angular_velocity = np.abs(
                compute_angular_velocity(
                    meta_trajectories_inner_outer[day][(source, sink)]["inner"],
                    WINDOW_SIZE,
                )
            )
            angular_velocity_outer = angular_velocity = np.abs(
                compute_angular_velocity(
                    meta_trajectories_inner_outer[day][(source, sink)]["outer"],
                    WINDOW_SIZE,
                )
            )

            ratio = angular_velocity_outer / angular_velocity_inner

            mask_outer_faster = angular_velocity_outer > angular_velocity_inner

            traj_com = meta_trajectories_inner_outer[day][(source, sink)]["com"]
            traj_com_outer_faster = traj_com[mask_outer_faster]
            traj_com_inner_faster = traj_com[~mask_outer_faster]

            ax[0].scatter(
                traj_com_outer_faster[:, 1],
                traj_com_outer_faster[:, 2],
                label="Outer faster",
                s=10,
                alpha=1,
                color="red",
            )

            ax[0].scatter(
                traj_com_inner_faster[:, 1],
                traj_com_inner_faster[:, 2],
                label="Inner faster",
                s=10,
                alpha=1,
                color="blue",
            )

            ax[0].set_xlabel("X [mm]")
            ax[0].set_ylabel("Y [mm]")
            ax[0].legend()
            ax[0].set_aspect("equal")

            sc_ratio = ax[1].scatter(
                traj_com[:, 1],
                traj_com[:, 2],
                c=ratio,
                s=10,
                alpha=1,
                cmap="viridis",
                vmin=0,
                vmax=10,
            )

            ax[1].set_title(f"Day {day}, {source} - {sink}")
            ax[1].set_xlabel("X [mm]")
            ax[1].set_ylabel("Y [mm]")
            ax[1].legend()
            ax[1].set_aspect("equal")

            fig.colorbar(sc_ratio, ax=ax[1], label="Ratio outer/inner")

            plt.tight_layout()
            plt.savefig(
                f"../data/figures/02_10_trajectories_inner_outer_faster/{day}_{source}_{sink}_trajectory.pdf"
            )
            plt.close()

    for day in ["06", "08"]:
        for source, sink in tqdm(meta_trajectories[day].keys()):

            if TURN_TYPE_DIAMOR[day][(source, sink)] == "straight":
                continue

            # for individual and dyads
            for size in [1, 2]:

                if size not in meta_trajectories[day][(source, sink)]:
                    continue

                meta_trajectory_size = meta_trajectories[day][(source, sink)][size]
                if len(meta_trajectory_size) < WINDOW_SIZE:
                    continue
                velocity, curvature = compute_curvature(
                    meta_trajectory_size, WINDOW_SIZE
                )
                t = np.linspace(0, 1, len(velocity))

                curvature = np.abs(curvature) * 1000
                velocity = velocity / 1000

                # portion of trajectory in straight part
                straight_portion = (meta_trajectory_size[:, 1] > TURNING_X_MIN) & (
                    meta_trajectory_size[:, 1] < TURNING_X_MAX
                )

                # turning portion
                turning_portion = meta_trajectory_size[:, 1] < TURNING_X_MIN

                df_values["id"].extend([f"{day}_{source}_{sink}"] * len(velocity))
                df_values["type"].extend([f"{size}"] * len(velocity))
                df_values["time_step"].extend(np.arange(len(velocity)))
                df_values["distance"].extend([None] * len(velocity))

                df_values["turning"].extend(turning_portion.tolist())
                df_values["straight"].extend(straight_portion.tolist())

                df_values["velocity"].extend(velocity.tolist())
                df_values["curvature"].extend(curvature.tolist())

                df_values["angular_velocity_from_curvature"].extend(
                    (velocity / (1 / curvature)).tolist()
                )

                angular_velocity = np.abs(
                    compute_angular_velocity(meta_trajectory_size, WINDOW_SIZE)
                )

                df_values["angular_velocity"].extend(angular_velocity.tolist())

                x = meta_trajectory_size[:, 1] / 1000
                y = meta_trajectory_size[:, 2] / 1000

                df_values["x"].extend(x.tolist())
                df_values["y"].extend(y.tolist())

    df = pd.DataFrame(df_values)
    df.to_csv("../data/intermediate/02_10_curvature_inner_outer.csv", index=False)
