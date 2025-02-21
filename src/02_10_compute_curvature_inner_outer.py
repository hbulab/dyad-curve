import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.stats import linregress

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

from parameters import STRONG_TURN_PAIRS_DIAMOR, WINDOW_SIZE, TURN_TYPE_DIAMOR, N_BINS


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

    curvatures = {"inner": [], "outer": [], "com": [], 1: [], 2: []}
    velocities = {"inner": [], "outer": [], "com": [], 1: [], 2: []}
    angular_velocities = {"inner": [], "outer": [], "com": [], 1: [], 2: []}
    angular_velocities_from_curvature = {
        "inner": [],
        "outer": [],
        "com": [],
        1: [],
        2: [],
    }
    distances = []

    for day in ["06", "08"]:
        for source, sink in tqdm(meta_trajectories[day].keys()):

            if TURN_TYPE_DIAMOR[day][(source, sink)] == "straight":
                continue

            fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

            for pos, color in zip(["inner", "outer", "com"], ["blue", "red", "black"]):

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

                curvature = np.abs(curvature) * 1000
                velocity = velocity / 1000

                velocities[pos].extend(velocity.tolist())
                curvatures[pos].extend(curvature.tolist())

                angular_velocities_from_curvature[pos].extend(
                    (velocity / (1 / curvature)).tolist()
                )

                angular_velocity = np.abs(
                    compute_angular_velocity(meta_trajectory_position, WINDOW_SIZE)
                )
                angular_velocities[pos].extend(angular_velocity.tolist())

                if pos == "com":
                    distances.extend(
                        [interpersonal_distances[day][(source, sink)]] * len(velocity)
                    )

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

            fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True)

            for pos, m in zip(["inner", "outer", "com"], ["o", "s", "D"]):

                if pos not in meta_trajectories[day][(source, sink)]:
                    continue

                meta_trajectory_position = meta_trajectories[day][(source, sink)][pos]
                if len(meta_trajectory_position) < WINDOW_SIZE:
                    continue
                velocity, curvature = compute_curvature(
                    meta_trajectory_position, WINDOW_SIZE
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

            # Add colorbars
            fig.colorbar(sc_curvature, ax=ax[0], label="Curvature [1/m]")
            fig.colorbar(sc_velocity, ax=ax[1], label="Velocity [m/s]")

            plt.tight_layout()
            plt.savefig(
                f"../data/figures/02_10_trajectories_with_velocity_curvature_inner_outer/{day}_{source}_{sink}_trajectory.pdf"
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

                velocities[size].extend(velocity.tolist())
                curvatures[size].extend(curvature.tolist())

                angular_velocities_from_curvature[size].extend(
                    (velocity / (1 / curvature)).tolist()
                )

                angular_velocity = np.abs(
                    compute_angular_velocity(meta_trajectory_size, WINDOW_SIZE)
                )
                angular_velocities[size].extend(angular_velocity.tolist())

    for pos in ["inner", "outer", "com", 1, 2]:
        velocities[pos] = np.array(velocities[pos])
        curvatures[pos] = np.array(curvatures[pos])
        angular_velocities[pos] = np.array(angular_velocities[pos])
        angular_velocities_from_curvature[pos] = np.array(
            angular_velocities_from_curvature[pos]
        )
        print(pos, np.mean(velocities[pos]), np.mean(curvatures[pos]))

    distances = np.array(distances) / 1000

    # plot binned velocity vs curvature

    fig, ax = plt.subplots(figsize=(12, 6))

    for pos, color in zip(["inner", "outer", 1, 2], ["blue", "red", "green", "orange"]):

        bin_centers, _, means, stds, errors, n_values = compute_binned_values(
            curvatures[pos], velocities[pos], n_bins=N_BINS, equal_frequency=True
        )

        ax.plot(bin_centers, means, color=color, label=f"{pos}")
        ax.fill_between(
            bin_centers,
            means - stds,
            means + stds,
            color=color,
            alpha=0.3,
        )
        ax.errorbar(
            bin_centers,
            means,
            yerr=errors,
            fmt="o",
            color=color,
            capsize=2,
        )

    ax.set_xlabel("$\\kappa$ [1/m]")
    ax.set_ylabel("Velocity [m/s]")

    ax.legend()
    ax.grid(color="gray", linestyle="--", linewidth=0.5)

    plt.savefig(
        "../data/figures/02_10_velocity_curvature_binned_velocity_inner_outer.pdf"
    )
    plt.close()

    # plot inner velocity vs outer velocity

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(velocities["inner"], velocities["outer"], s=10, alpha=0.5)

    # fit piecewise linear
    params, r_squared = fit_piecewise_linear(velocities["inner"], velocities["outer"])
    plot_piecewise_linear(0.5, 1.4, *params, ax, label=f"$R^2 = {r_squared:.2f}$")

    ax.plot([0.5, 1.4], [0.5, 1.4], color="black", linestyle="--")

    ax.set_xlabel("Inner velocity [m/s]")
    ax.set_ylabel("Outer velocity [m/s]")
    ax.set_aspect("equal")
    ax.grid(color="gray", linestyle="--", linewidth=0.5)

    ax.legend()

    plt.savefig("../data/figures/02_10_inner_outer_velocity.pdf")
    plt.close()

    # inner curvature vs outer curvature

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(curvatures["inner"], curvatures["outer"], s=10, alpha=0.5)

    # linear regression
    slope, intercept, r_value, p_value, std_err = linregress(
        curvatures["inner"], curvatures["outer"]
    )
    x = np.linspace(0, 0.9, 100)
    y = slope * x + intercept
    ax.plot(
        x,
        y,
        color="red",
        label=f"y = {slope:.2f}x + {intercept:.2f} ($R^2 = {r_value**2:.2f}$)",
    )

    ax.plot([0, 0.9], [0, 0.9], color="black", linestyle="--")

    ax.set_xlabel("Inner curvature [1/m]")
    ax.set_ylabel("Outer curvature [1/m]")
    ax.set_aspect("equal")
    ax.grid(color="gray", linestyle="--", linewidth=0.5)

    ax.legend()

    plt.savefig("../data/figures/02_10_inner_outer_curvature.pdf")
    plt.close()

    # plot binned log velocity vs log curvature

    fig, ax = plt.subplots(figsize=(12, 6))

    for pos, color in zip(["inner", "outer", 1, 2], ["blue", "red", "green", "orange"]):

        bin_centers, _, means, stds, errors, n_values = compute_binned_values(
            np.log(curvatures[pos]),
            np.log(velocities[pos]),
            # velocities[pos],
            n_bins=N_BINS,
            # equal_frequency=True,
        )

        ax.plot(bin_centers, means, color=color, label=f"{pos}")
        ax.fill_between(
            bin_centers,
            means - stds,
            means + stds,
            color=color,
            alpha=0.3,
        )
        ax.errorbar(
            bin_centers,
            means,
            yerr=errors,
            fmt="o",
            color=color,
            capsize=2,
        )

    ax.set_xlabel("$\\log(\\kappa)$")
    ax.set_ylabel("$\\log(v)$")

    ax.legend()
    ax.grid(color="gray", linestyle="--", linewidth=0.5)

    plt.savefig("../data/figures/02_10_velocity_curvature_binned_log_velocity.pdf")

    # plot binned log velocity vs log radius

    fig, ax = plt.subplots(figsize=(12, 6))

    for pos, color in zip(["inner", "outer", 1, 2], ["blue", "red", "green", "orange"]):

        bin_centers, _, means, stds, errors, n_values = compute_binned_values(
            np.log(1 / curvatures[pos]),
            np.log(velocities[pos]),
            # velocities[pos],
            n_bins=N_BINS,
            equal_frequency=True,
        )

        ax.plot(bin_centers, means, color=color, label=f"{pos}")
        ax.fill_between(
            bin_centers,
            means - stds,
            means + stds,
            color=color,
            alpha=0.3,
        )
        ax.errorbar(
            bin_centers,
            means,
            yerr=errors,
            fmt="o",
            color=color,
            capsize=2,
        )

    ax.set_xlabel("$\\log(R)$")
    ax.set_ylabel("$\\log(v)$")

    ax.legend()
    ax.grid(color="gray", linestyle="--", linewidth=0.5)

    plt.savefig(
        "../data/figures/02_10_velocity_curvature_binned_log_velocity_log_radius.pdf"
    )

    # plot binned difference in velocity vs curvature

    fig, ax = plt.subplots(figsize=(12, 6))

    d_v = velocities["inner"] - velocities["outer"]

    bin_centers, _, means, stds, errors, n_values = compute_binned_values(
        curvatures["com"], d_v, n_bins=N_BINS, equal_frequency=True
    )

    ax.plot(bin_centers, means, color="black")
    ax.fill_between(
        bin_centers,
        means - stds,
        means + stds,
        color="black",
        alpha=0.3,
    )
    ax.errorbar(
        bin_centers,
        means,
        yerr=errors,
        fmt="o",
        color="black",
        capsize=2,
    )

    ax.set_xlabel("$\\kappa$ [1/m]")
    ax.set_ylabel("$\\Delta v$ [m/s]")
    ax.grid(color="gray", linestyle="--", linewidth=0.5)

    plt.savefig(
        "../data/figures/02_10_velocity_curvature_binned_difference_velocity.pdf"
    )

    # plot binned difference in velocity vs log(curvature)

    fig, ax = plt.subplots(figsize=(12, 6))

    d_v = velocities["inner"] - velocities["outer"]

    bin_centers, _, means, stds, errors, n_values = compute_binned_values(
        np.log(curvatures["com"]),
        d_v,
        n_bins=N_BINS,
        equal_frequency=True,
    )

    ax.plot(bin_centers, means, color="black")
    ax.fill_between(
        bin_centers,
        means - stds,
        means + stds,
        color="black",
        alpha=0.3,
    )
    ax.errorbar(
        bin_centers,
        means,
        yerr=errors,
        fmt="o",
        color="black",
        capsize=2,
    )

    ax.set_xlabel("$\\log(\\kappa)$")
    ax.set_ylabel("$\\Delta v$ [m/s]")

    ax.grid(color="gray", linestyle="--", linewidth=0.5)

    plt.savefig(
        "../data/figures/02_10_velocity_curvature_binned_difference_velocity_log_curvature.pdf"
    )

    # curvature outer/inner expected
    radius_inner = 1 / curvatures["inner"]
    radius_outer = 1 / curvatures["outer"]

    radius_outer_expected = radius_inner + distances
    curvature_outer_expected = 1 / radius_outer_expected

    # linear regression
    slope, intercept, r_value, p_value, std_err = linregress(
        curvatures["inner"], curvatures["outer"]
    )
    x = np.linspace(0, 0.7, 100)
    y = slope * x + intercept

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        curvatures["inner"], curvatures["outer"], s=10, alpha=0.5, label="observed"
    )
    ax.scatter(
        curvatures["inner"], curvature_outer_expected, s=10, alpha=0.5, label="expected"
    )

    ax.plot(
        x,
        y,
        color="red",
        label=f"y = {slope:.2f}x + {intercept:.2f} ($R^2 = {r_value**2:.2f}$)",
    )

    ax.plot([0, 0.7], [0, 0.7], color="black", linestyle="--")

    ax.set_xlabel("Inner curvature [1/m]")
    ax.set_ylabel("Outer curvature [1/m]")

    ax.set_aspect("equal")

    ax.legend()
    ax.grid(color="gray", linestyle="--", linewidth=0.5)

    plt.savefig("../data/figures/02_10_curvature_inner_outer_expected.pdf")
    plt.close()

    # plot angular velocity outer/inner

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        angular_velocities["inner"],
        angular_velocities["outer"],
        s=10,
        alpha=0.5,
        label="computed ($\\omega = \\frac{d\\theta}{dt}$)",
    )
    ax.scatter(
        angular_velocities_from_curvature["inner"],
        angular_velocities_from_curvature["outer"],
        s=10,
        alpha=0.5,
        label="from curvature ($\\omega = \\frac{v}{R}$)",
    )

    ax.plot([0, 0.4], [0, 0.4], color="black", linestyle="--")

    ax.set_xlabel("Inner angular velocity [rad/s]")
    ax.set_ylabel("Outer angular velocity [rad/s]")

    ax.set_aspect("equal")
    ax.legend()
    ax.grid(color="gray", linestyle="--", linewidth=0.5)

    plt.savefig("../data/figures/02_10_angular_velocity_inner_outer.pdf")
    plt.close()

    # plot log binned angular velocity outer/inner

    fig, ax = plt.subplots(figsize=(12, 6))

    for angular_velocities_type, label, color in zip(
        [
            angular_velocities,
            angular_velocities_from_curvature,
        ],
        [
            "computed ($\\omega = \\frac{d\\theta}{dt}$)",
            "from curvature ($\\omega = \\frac{v}{R}$)",
        ],
        ["blue", "orange"],
    ):

        bin_centers, _, means, stds, errors, n_values = compute_binned_values(
            np.log(angular_velocities_type["inner"]),
            np.log(angular_velocities["outer"]),
            n_bins=N_BINS,
            equal_frequency=False,
        )

        ax.plot(bin_centers, means, label=label, color=color)

        ax.fill_between(
            bin_centers,
            means - stds,
            means + stds,
            alpha=0.3,
            color=color,
        )
        ax.errorbar(
            bin_centers,
            means,
            yerr=errors,
            fmt="o",
            capsize=2,
            color=color,
        )

    # add diagonal
    ax.plot([-14, 0], [-14, 0], color="black", linestyle="--")

    ax.set_xlabel("$\\log(\\omega_{inner})$")
    ax.set_ylabel("$\\log(\\omega_{outer})$")

    ax.set_aspect("equal")

    ax.legend()

    ax.grid(color="gray", linestyle="--", linewidth=0.5)

    plt.savefig("../data/figures/02_10_angular_velocity_inner_outer_binned.pdf")
    plt.close()

    # print sorted interpersonal_distances
    for day in ["06", "08"]:
        print(f"Day {day}")
        sorted_distances = sorted(
            interpersonal_distances[day].items(), key=lambda x: x[1], reverse=True
        )
        for (source, sink), distance in sorted_distances:
            print(f"{source} - {sink}: {distance:.2f}")
        print(f"Mean: {np.mean(list(interpersonal_distances[day].values())):.2f}")
