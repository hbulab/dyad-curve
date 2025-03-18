import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

from utils import (
    compute_binned_values,
    fit_piecewise_linear,
    plot_piecewise_linear,
)

import scienceplots

plt.style.use("science")


from parameters import N_BINS, CURVATURE_THRESHOLD

if __name__ == "__main__":

    data_inner_outer = pd.read_csv(
        "../data/intermediate/02_10_curvature_inner_outer.csv"
    )

    data_inner_outer["log_curvature"] = np.log(data_inner_outer["curvature"])
    data_inner_outer_no_com = data_inner_outer[(data_inner_outer["type"] != "com")]

    # find data points where curvature of com
    high_curvature_pairs = data_inner_outer.loc[
        (data_inner_outer["type"] == "com")
        & (data_inner_outer["curvature"] > CURVATURE_THRESHOLD),
        ["id", "time_step"],
    ]
    data_inner_outer_high_curvature = data_inner_outer.merge(
        high_curvature_pairs, on=["id", "time_step"]
    )

    # TEST PLOT
    trajectory_ids = data_inner_outer_high_curvature["id"].unique()

    n_trajectories = len(trajectory_ids)
    n_rows = 2
    n_cols = int(np.ceil(n_trajectories / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6))
    axes = axes.flatten()
    for i, trajectory_id in enumerate(trajectory_ids):
        for type in ["inner", "outer", "1"]:  # Use actual types in the dataset
            subset = data_inner_outer_high_curvature[
                (data_inner_outer_high_curvature["id"] == trajectory_id)
                & (data_inner_outer_high_curvature["type"] == type)
            ]
            axes[i].scatter(subset["x"], subset["y"], label=type, alpha=0.5, s=3)

        axes[i].set_title(f"Trajectory {trajectory_id}")
        axes[i].legend()
        axes[i].set_aspect("equal")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    plt.close()

    # plot histogram of curvature
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    sns.histplot(
        data_inner_outer_no_com,
        x="log_curvature",
        bins=30,
        hue="type",
        kde=True,
        ax=ax[0],
        alpha=0.5,
        binrange=(-12, 0),
        stat="density",
    )

    sns.histplot(
        data_inner_outer_no_com[data_inner_outer_no_com["straight"] == True],
        x="log_curvature",
        bins=30,
        hue="type",
        kde=True,
        ax=ax[1],
        alpha=0.5,
        binrange=(-12, 0),
        stat="density",
    )

    sns.histplot(
        data_inner_outer_no_com[data_inner_outer_no_com["turning"] == True],
        x="log_curvature",
        bins=30,
        hue="type",
        kde=True,
        ax=ax[2],
        alpha=0.5,
        binrange=(-12, 0),
        stat="density",
    )

    ax[0].set_title("All")
    ax[1].set_title("Straight")
    ax[2].set_title("Turning")

    plt.tight_layout()
    plt.savefig("../data/figures/02_11_curvature_histogram.pdf")
    plt.close()

    # plot binned velocity vs curvature
    fig, ax = plt.subplots(figsize=(12, 6))

    for pos, color in zip(
        ["inner", "outer", "1", "2"], ["blue", "red", "green", "orange"]
    ):

        bin_centers, _, means, stds, errors, _, _, _, n_values = compute_binned_values(
            data_inner_outer_high_curvature[
                data_inner_outer_high_curvature["type"] == pos
            ]["curvature"],
            data_inner_outer_high_curvature[
                data_inner_outer_high_curvature["type"] == pos
            ]["velocity"],
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

    ax.set_xlabel("$\\kappa$ [1/m]")
    ax.set_ylabel("Velocity [m/s]")

    ax.legend()
    ax.grid(color="gray", linestyle="--", linewidth=0.5)

    plt.savefig(
        "../data/figures/02_11_velocity_curvature_binned_velocity_inner_outer.pdf"
    )
    plt.close()

    # plot inner velocity vs outer velocity

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "inner"
        ]["velocity"],
        data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "outer"
        ]["velocity"],
        s=10,
        alpha=0.5,
    )

    # fit piecewise linear
    params, r_squared = fit_piecewise_linear(
        data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "inner"
        ]["velocity"],
        data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "outer"
        ]["velocity"],
    )
    plot_piecewise_linear(0.5, 1.4, *params, ax, label=f"$R^2 = {r_squared:.2f}$")

    ax.plot([0.5, 1.4], [0.5, 1.4], color="black", linestyle="--")

    ax.set_xlabel("Inner velocity [m/s]")
    ax.set_ylabel("Outer velocity [m/s]")
    ax.set_aspect("equal")
    ax.grid(color="gray", linestyle="--", linewidth=0.5)

    ax.legend()

    plt.savefig("../data/figures/02_11_inner_outer_velocity.pdf")
    plt.close()

    # inner curvature vs outer curvature

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "inner"
        ]["curvature"],
        data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "outer"
        ]["curvature"],
        s=10,
        alpha=0.5,
    )

    # linear regression
    slope, intercept, r_value, p_value, std_err = linregress(
        data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "inner"
        ]["curvature"],
        data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "outer"
        ]["curvature"],
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

    plt.savefig("../data/figures/02_11_inner_outer_curvature.pdf")
    plt.close()

    # plot binned log velocity vs log curvature

    fig, ax = plt.subplots(figsize=(12, 6))

    for pos, color in zip(
        ["inner", "outer", "1", "2"], ["blue", "red", "green", "orange"]
    ):

        bin_centers, _, means, stds, errors, _, _, _, n_values = compute_binned_values(
            data_inner_outer_high_curvature[
                data_inner_outer_high_curvature["type"] == pos
            ]["log_curvature"],
            np.log(
                data_inner_outer_high_curvature[
                    data_inner_outer_high_curvature["type"] == pos
                ]["velocity"]
            ),
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

    plt.savefig("../data/figures/02_11_velocity_curvature_binned_log_velocity.pdf")

    # plot binned log velocity vs log radius

    fig, ax = plt.subplots(figsize=(12, 6))

    for pos, color in zip(
        ["inner", "outer", "1", "2"], ["blue", "red", "green", "orange"]
    ):

        bin_centers, _, means, stds, errors, _, _, _, n_values = compute_binned_values(
            np.log(
                1
                / data_inner_outer_high_curvature[
                    data_inner_outer_high_curvature["type"] == pos
                ]["curvature"]
            ),
            np.log(
                data_inner_outer_high_curvature[
                    data_inner_outer_high_curvature["type"] == pos
                ]["velocity"]
            ),
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
        "../data/figures/02_11_velocity_curvature_binned_log_velocity_log_radius.pdf"
    )

    # plot binned difference in velocity vs curvature

    fig, ax = plt.subplots(figsize=(12, 6))

    d_v = (
        data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "inner"
        ]["velocity"].values
        - data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "outer"
        ]["velocity"].values
    )

    bin_centers, _, means, stds, errors, _, _, _, n_values = compute_binned_values(
        data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "com"
        ]["curvature"],
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

    ax.set_xlabel("$\\kappa$ [1/m]")
    ax.set_ylabel("$\\Delta v$ [m/s]")
    ax.grid(color="gray", linestyle="--", linewidth=0.5)

    plt.savefig(
        "../data/figures/02_11_velocity_curvature_binned_difference_velocity.pdf"
    )

    # plot binned difference in velocity vs log(curvature)

    fig, ax = plt.subplots(figsize=(12, 6))

    d_v = (
        data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "inner"
        ]["velocity"].values
        - data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "outer"
        ]["velocity"].values
    )

    bin_centers, _, means, stds, errors, _, _, _, n_values = compute_binned_values(
        np.log(
            data_inner_outer_high_curvature[
                data_inner_outer_high_curvature["type"] == "com"
            ]["curvature"]
        ),
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
        "../data/figures/02_11_velocity_curvature_binned_difference_velocity_log_curvature.pdf"
    )

    # curvature outer/inner expected
    radius_inner = (
        1
        / data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "inner"
        ]["curvature"]
    )
    radius_outer = (
        1
        / data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "outer"
        ]["curvature"]
    )

    radius_outer_expected = (
        radius_inner
        + data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "inner"
        ]["distance"]
        / 1000
    )
    curvature_outer_expected = 1 / radius_outer_expected

    # linear regression
    slope, intercept, r_value, p_value, std_err = linregress(
        data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "inner"
        ]["curvature"],
        data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "outer"
        ]["curvature"],
    )
    x = np.linspace(0, 0.7, 100)
    y = slope * x + intercept

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "inner"
        ]["curvature"],
        data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "outer"
        ]["curvature"],
        s=10,
        alpha=0.5,
        label="observed",
    )
    ax.scatter(
        data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "inner"
        ]["curvature"],
        curvature_outer_expected,
        s=10,
        alpha=0.5,
        label="expected",
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

    plt.savefig("../data/figures/02_11_curvature_inner_outer_expected.pdf")
    plt.close()

    # plot angular velocity outer/inner

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "inner"
        ]["angular_velocity"],
        data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "outer"
        ]["angular_velocity"],
        s=10,
        alpha=0.5,
        label="computed ($\\omega = \\frac{d\\theta}{dt}$)",
    )
    ax.scatter(
        data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "inner"
        ]["angular_velocity_from_curvature"],
        data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "outer"
        ]["angular_velocity_from_curvature"],
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

    plt.savefig("../data/figures/02_11_angular_velocity_inner_outer.pdf")
    plt.close()

    # plot log binned angular velocity outer/inner

    fig, ax = plt.subplots(figsize=(12, 6))

    for angular_velocities_type, label, color in zip(
        ["angular_velocity", "angular_velocity_from_curvature"],
        [
            "computed ($\\omega = \\frac{d\\theta}{dt}$)",
            "from curvature ($\\omega = \\frac{v}{R}$)",
        ],
        ["blue", "orange"],
    ):

        (
            bin_centers,
            _,
            means_outer,
            stds_outer,
            errors_outer,
            means_inner,
            stds_inner,
            errors_inner,
            n_values,
        ) = compute_binned_values(
            np.log(
                data_inner_outer_high_curvature[
                    data_inner_outer_high_curvature["type"] == "inner"
                ][angular_velocities_type]
            ),
            np.log(
                data_inner_outer_high_curvature[
                    data_inner_outer_high_curvature["type"] == "outer"
                ][angular_velocities_type]
            ),
            n_bins=8,
            equal_frequency=False,
        )

        ax.errorbar(
            means_inner,
            means_outer,
            yerr=errors_outer,
            xerr=errors_inner,
            fmt="-o",
            capsize=2,
            color=color,
        )

        ax.scatter(
            np.log(
                data_inner_outer_high_curvature[
                    data_inner_outer_high_curvature["type"] == "inner"
                ][angular_velocities_type]
            ),
            np.log(
                data_inner_outer_high_curvature[
                    data_inner_outer_high_curvature["type"] == "outer"
                ][angular_velocities_type]
            ),
            s=10,
            alpha=0.5,
            color=color,
        )

    # add diagonal
    ax.plot([-14, 0], [-14, 0], color="black", linestyle="--")

    ax.set_xlabel("$\\log(\\omega_{inner})$")
    ax.set_ylabel("$\\log(\\omega_{outer})$")

    ax.set_aspect("equal")

    ax.grid(color="gray", linestyle="--", linewidth=0.5)

    plt.savefig("../data/figures/02_11_angular_velocity_inner_outer_binned.pdf")
    plt.close()

    # plot log binned angular velocity inner/outer

    fig, ax = plt.subplots(figsize=(12, 6))

    for angular_velocities_type, label, color in zip(
        [
            "angular_velocity",
            "angular_velocities_from_curvature",
        ],
        [
            "computed ($\\omega = \\frac{d\\theta}{dt}$)",
            "from curvature ($\\omega = \\frac{v}{R}$)",
        ],
        ["blue", "orange"],
    ):

        (
            bin_centers,
            _,
            means_inner,
            stds_inner,
            errors_inner,
            means_outer,
            stds_outer,
            errors_outer,
            n_values,
        ) = compute_binned_values(
            np.log(
                data_inner_outer_high_curvature[
                    data_inner_outer_high_curvature["type"] == "outer"
                ]["angular_velocity"]
            ),
            np.log(
                data_inner_outer_high_curvature[
                    data_inner_outer_high_curvature["type"] == "inner"
                ]["angular_velocity"]
            ),
            n_bins=8,
            equal_frequency=False,
            min_v=-14,
            max_v=0,
        )

        ax.errorbar(
            means_outer,
            means_inner,
            yerr=errors_inner,
            xerr=errors_outer,
            fmt="-o",
            capsize=2,
            color=color,
        )

        ax.scatter(
            np.log(
                data_inner_outer_high_curvature[
                    data_inner_outer_high_curvature["type"] == "outer"
                ]["angular_velocity"]
            ),
            np.log(
                data_inner_outer_high_curvature[
                    data_inner_outer_high_curvature["type"] == "inner"
                ]["angular_velocity"]
            ),
            s=10,
            alpha=0.5,
            color=color,
        )

    # add diagonal
    ax.plot([-14, 0], [-14, 0], color="black", linestyle="--")

    ax.set_xlabel("$\\log(\\omega_{outer})$")
    ax.set_ylabel("$\\log(\\omega_{inner})$")
    ax.set_aspect("equal")

    ax.grid(color="gray", linestyle="--", linewidth=0.5)

    plt.savefig("../data/figures/02_11_angular_velocity_outer_inner_binned.pdf")

    # # plot log binned angular velocity inner/outer for high curvature

    # fig, ax = plt.subplots(figsize=(12, 6))

    # mask = data_inner_outer_high_curvature["curvature"] > CURVATURE_THRESHOLD

    # (
    #     bin_centers,
    #     _,
    #     means_outer,
    #     stds_outer,
    #     errors_outer,
    #     means_inner,
    #     stds_inner,
    #     errors_innner,
    #     n_values,
    # ) = compute_binned_values(
    #     np.log(
    #         data_inner_outer_high_curvature[data_inner_outer_high_curvature["type"] == "inner"][mask][
    #             "angular_velocity"
    #         ]
    #     ),
    #     np.log(
    #         data_inner_outer_high_curvature[data_inner_outer_high_curvature["type"] == "outer"][mask][
    #             "angular_velocity"
    #         ]
    #     ),
    #     n_bins=8,
    #     equal_frequency=False,
    # )

    # ax.errorbar(
    #     means_inner,
    #     means_outer,
    #     yerr=errors_outer,
    #     xerr=errors_innner,
    #     fmt="-",
    #     capsize=2,
    #     color="red",
    #     label=label,
    # )

    # ax.scatter(
    #     np.log(
    #         data_inner_outer_high_curvature[data_inner_outer_high_curvature["type"] == "inner"][mask][
    #             "angular_velocity_from_curvature"
    #         ]
    #     ),
    #     np.log(
    #         data_inner_outer_high_curvature[data_inner_outer_high_curvature["type"] == "outer"][mask][
    #             "angular_velocity_from_curvature"
    #         ]
    #     ),
    #     s=10,
    #     alpha=0.2,
    #     color="blue",
    # )

    # # add diagonal
    # ax.plot(
    #     [np.log(CURVATURE_THRESHOLD), 0],
    #     [np.log(CURVATURE_THRESHOLD), 0],
    #     color="black",
    #     linestyle="--",
    # )

    # ax.set_xlabel("$\\log(\\omega_{inner})$")
    # ax.set_ylabel("$\\log(\\omega_{outer})$")
    # ax.set_aspect("equal")

    # ax.legend()
    # ax.grid(color="gray", linestyle="--", linewidth=0.5)

    # plt.savefig(
    #     "../data/figures/02_11_angular_velocity_outer_inner_binned_high_curvature.pdf"
    # )
    # plt.close()

    # # plot log binned angular velocity inner/outer for high curvature

    # fig, ax = plt.subplots(figsize=(12, 6))

    # mask = data_inner_outer_high_curvature["curvature"] > CURVATURE_THRESHOLD

    # (
    #     bin_centers,
    #     _,
    #     means_inner,
    #     stds_inner,
    #     errors_inner,
    #     means_outer,
    #     stds_outer,
    #     errors_outer,
    #     n_values,
    # ) = compute_binned_values(
    #     np.log(
    #         data_inner_outer_high_curvature[data_inner_outer_high_curvature["type"] == "outer"][mask][
    #             "angular_velocity_from_curvature"
    #         ]
    #     ),
    #     np.log(
    #         data_inner_outer_high_curvature[data_inner_outer_high_curvature["type"] == "inner"][mask][
    #             "angular_velocity_from_curvature"
    #         ]
    #     ),
    #     n_bins=8,
    #     equal_frequency=False,
    # )

    # ax.errorbar(
    #     means_outer,
    #     means_inner,
    #     yerr=errors_inner,
    #     xerr=errors_outer,
    #     fmt="-",
    #     capsize=2,
    #     color="red",
    #     label=label,
    # )

    # ax.scatter(
    #     np.log(
    #         data_inner_outer_high_curvature[data_inner_outer_high_curvature["type"] == "outer"][mask][
    #             "angular_velocity_from_curvature"
    #         ]
    #     ),
    #     np.log(
    #         data_inner_outer_high_curvature[data_inner_outer_high_curvature["type"] == "inner"][mask][
    #             "angular_velocity_from_curvature"
    #         ]
    #     ),
    #     s=10,
    #     alpha=0.2,
    #     color="blue",
    # )

    # # add diagonal
    # ax.plot(
    #     [np.log(CURVATURE_THRESHOLD), 0],
    #     [np.log(CURVATURE_THRESHOLD), 0],
    #     color="black",
    #     linestyle="--",
    # )

    # ax.set_xlabel("$\\log(\\omega_{outer})$")
    # ax.set_ylabel("$\\log(\\omega_{inner})$")
    # ax.set_aspect("equal")

    # ax.legend()
    # ax.grid(color="gray", linestyle="--", linewidth=0.5)

    # plt.savefig(
    #     "../data/figures/02_11_angular_velocity_inner_outer_binned_high_curvature.pdf"
    # )
    # plt.close()

    # plot binned angular velocity inner/outer rotated by 45 degrees

    fig, ax = plt.subplots(figsize=(12, 6))

    for angular_velocities_type, label, color in zip(
        [
            "angular_velocity",
            "angular_velocity_from_curvature",
        ],
        [
            "computed ($\\omega = \\frac{d\\theta}{dt}$)",
            "from curvature ($\\omega = \\frac{v}{R}$)",
        ],
        ["blue", "orange"],
    ):

        x = np.log(
            data_inner_outer_high_curvature[
                data_inner_outer_high_curvature["type"] == "inner"
            ][angular_velocities_type].values
        )
        y = np.log(
            data_inner_outer_high_curvature[
                data_inner_outer_high_curvature["type"] == "outer"
            ][angular_velocities_type].values
        )

        center = np.array([0, 0])
        angle = -np.pi / 4

        x_rot = np.cos(angle) * x - np.sin(angle) * y
        y_rot = np.sin(angle) * x + np.cos(angle) * y

        (
            bin_centers,
            _,
            means_rot,
            stds_rot,
            errors_rot,
            _,
            _,
            _,
            n_values,
        ) = compute_binned_values(
            x_rot,
            y_rot,
            n_bins=8,
            equal_frequency=False,
        )

        ax.errorbar(
            bin_centers,
            means_rot,
            yerr=errors_rot,
            fmt="-o",
            capsize=2,
            color=color,
            label=label,
        )

        ax.scatter(
            x_rot,
            y_rot,
            s=10,
            alpha=0.5,
            color=color,
        )

    plt.xlabel("$\\log(\\omega_{inner})$")
    plt.ylabel("$\\log(\\omega_{outer})$")

    plt.legend()
    plt.grid(color="gray", linestyle="--", linewidth=0.5)

    plt.savefig("../data/figures/02_11_angular_velocity_inner_outer_binned_rotated.pdf")
    plt.close()

    # plot  angular velocity vs log curvature (binned)

    fig, ax = plt.subplots(figsize=(8, 8))

    for pos, color in zip(["inner", "outer"], ["blue", "red"]):

        bin_centers, _, means, stds, errors, _, _, _, n_values = compute_binned_values(
            np.log(
                data_inner_outer_high_curvature[
                    data_inner_outer_high_curvature["type"] == pos
                ]["curvature"]
            ),
            np.log(
                data_inner_outer_high_curvature[
                    data_inner_outer_high_curvature["type"] == pos
                ]["angular_velocity"]
            ),
            n_bins=N_BINS,
            equal_frequency=False,
        )

        ax.plot(bin_centers, means, label=f"{pos}", color=color)

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

    ax.set_xlabel("$\\log(\\kappa)$")
    ax.set_ylabel("$\\omega_{inner}$ [rad/s]")
    ax.legend()
    ax.grid(color="gray", linestyle="--", linewidth=0.5)

    plt.savefig("../data/figures/02_11_angular_velocity_curvature_binned.pdf")
    plt.close()

    # print log(w_inner) on x axis and w_inner - w_outer on y axis
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].scatter(
        np.log(
            data_inner_outer_high_curvature[
                data_inner_outer_high_curvature["type"] == "inner"
            ]["angular_velocity"]
        ),
        data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "inner"
        ]["angular_velocity"].values
        - data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "outer"
        ]["angular_velocity"].values,
        s=10,
        alpha=0.5,
    )

    bin_centers, _, means, stds, errors, _, _, _, n_values = compute_binned_values(
        np.log(
            data_inner_outer_high_curvature[
                data_inner_outer_high_curvature["type"] == "inner"
            ]["angular_velocity"]
        ),
        data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "inner"
        ]["angular_velocity"].values
        - data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "outer"
        ]["angular_velocity"].values,
        n_bins=N_BINS,
        equal_frequency=False,
    )

    ax[0].plot(bin_centers, means, color="black")
    ax[0].fill_between(
        bin_centers,
        means - stds,
        means + stds,
        color="black",
        alpha=0.3,
    )
    ax[0].errorbar(
        bin_centers,
        means,
        yerr=errors,
        fmt="o",
        capsize=2,
        color="black",
    )

    ax[0].set_xlabel("$\\omega_{inner}$ [rad/s]")
    ax[0].set_ylabel("$\\omega_{inner} - \\omega_{outer}$ [rad/s]")

    ax[0].grid(color="gray", linestyle="--", linewidth=0.5)

    ax[1].scatter(
        np.log(
            data_inner_outer_high_curvature[
                data_inner_outer_high_curvature["type"] == "outer"
            ]["angular_velocity"]
        ),
        data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "inner"
        ]["angular_velocity"].values
        - data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "outer"
        ]["angular_velocity"].values,
        s=10,
        alpha=0.5,
    )

    bin_centers, _, means, stds, errors, _, _, _, n_values = compute_binned_values(
        np.log(
            data_inner_outer_high_curvature[
                data_inner_outer_high_curvature["type"] == "outer"
            ]["angular_velocity"]
        ),
        data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "inner"
        ]["angular_velocity"].values
        - data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "outer"
        ]["angular_velocity"].values,
        n_bins=N_BINS,
        equal_frequency=False,
    )

    ax[1].plot(bin_centers, means, color="black")
    ax[1].fill_between(
        bin_centers,
        means - stds,
        means + stds,
        color="black",
        alpha=0.3,
    )
    ax[1].errorbar(
        bin_centers,
        means,
        yerr=errors,
        fmt="o",
        capsize=2,
        color="black",
    )

    ax[1].set_xlabel("$\\omega_{outer}$ [rad/s]")
    ax[1].set_ylabel("$\\omega_{inner} - \\omega_{outer}$ [rad/s]")

    ax[1].grid(color="gray", linestyle="--", linewidth=0.5)

    plt.savefig("../data/figures/02_11_angular_velocity_difference.pdf")

    # print sorted interpersonal_distances
    print(
        data_inner_outer_high_curvature[
            data_inner_outer_high_curvature["type"] == "inner"
        ]
        .groupby("id")["distance"]
        .mean()
        .sort_values()
    )

    # for day in ["06", "08"]:
    #     print(f"Day {day}")
    #     sorted_distances = sorted(
    #         interpersonal_distances[day].items(), key=lambda x: x[1], reverse=True
    #     )
    #     for (source, sink), distance in sorted_distances:
    #         print(f"{source} - {sink}: {distance:.2f}")
    #     print(f"Mean: {np.mean(list(interpersonal_distances[day].values())):.2f}")
