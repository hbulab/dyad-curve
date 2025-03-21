from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from tqdm import tqdm

import scienceplots

plt.style.use("science")


from utils import (
    pickle_load,
    compute_binned_values,
    compute_curvature,
    fit_hinge_function,
    plot_hinge_function,
)

from parameters import WINDOW_SIZE, N_BINS


if __name__ == "__main__":
    meta_trajectories = pickle_load(
        "../data/intermediate/02_07_meta_trajectories_diamor.pkl"
    )

    curvatures = {size: [] for size in [1, 2, "all"]}
    velocities = {size: [] for size in [1, 2, "all"]}
    indexes = []

    for day in ["06", "08"]:

        for source, sink in tqdm(meta_trajectories[day].keys()):

            if "all" in meta_trajectories[day][(source, sink)]:
                if len(meta_trajectories[day][(source, sink)]["all"]) < WINDOW_SIZE:
                    continue

                velocity_all, curvature_all = compute_curvature(
                    meta_trajectories[day][(source, sink)]["all"], WINDOW_SIZE
                )
                curvature_all = np.abs(curvature_all) * 1000
                velocity_all = velocity_all / 1000

                curvatures["all"].extend(curvature_all.tolist())
                velocities["all"].extend(velocity_all.tolist())

            # plot velocity and curvature for each size
            fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

            for size in [1, 2]:

                if size not in meta_trajectories[day][(source, sink)]:
                    continue

                meta_trajectory_size = meta_trajectories[day][(source, sink)][size]
                if len(meta_trajectory_size) < WINDOW_SIZE:
                    print(len(meta_trajectory_size))
                velocity, curvature = compute_curvature(
                    meta_trajectory_size, WINDOW_SIZE
                )
                t = np.linspace(0, 1, len(velocity))

                curvature = np.abs(curvature) * 1000
                velocity = velocity / 1000

                velocities[size].extend(velocity.tolist())
                curvatures[size].extend(curvature.tolist())

                if size == 2:
                    indexes.extend([f"{day}_{source}_{sink}"] * len(velocity))

                axes[0].plot(
                    t,
                    velocity,
                    color="orange" if size == 1 else "blue",
                    label="Individual" if size == 1 else "Dyad",
                )
                axes[1].plot(
                    t,
                    curvature,
                    color="orange" if size == 1 else "blue",
                    label="Individual" if size == 1 else "Dyad",
                )

            axes[0].set_ylabel("$v$ [m/s]")
            axes[0].legend()
            axes[0].set_ylim(0.5, 1.5)
            axes[0].grid(color="gray", linestyle="--", linewidth=0.5)

            axes[1].set_xlabel("$t$")
            axes[1].set_ylabel("$\\kappa$ [1/m]")
            axes[1].legend()
            axes[1].set_ylim(-0.01, 1)
            axes[1].grid(color="gray", linestyle="--", linewidth=0.5)

            plt.savefig(
                f"../data/figures/02_08_velocity_curvature/{day}_{source}_{sink}.pdf"
            )
            plt.close()

            # plot trajectory with curvature and trajectory with velocity
            sc_curvature = None
            sc_velocity = None

            fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True, sharey=True)

            # Colormap for reference
            cmap_curvature = plt.get_cmap("viridis")
            cmap_velocity = plt.get_cmap("viridis")

            for size, m in zip([1, 2], ["o", "s"]):

                if size not in meta_trajectories[day][(source, sink)]:
                    continue

                meta_trajectory_size = meta_trajectories[day][(source, sink)][size]
                if len(meta_trajectory_size) < WINDOW_SIZE:
                    continue
                velocity, curvature = compute_curvature(
                    meta_trajectory_size, WINDOW_SIZE
                )

                curvature = np.abs(curvature) * 1000
                velocity = velocity / 1000

                sc_curvature = ax[0].scatter(
                    meta_trajectory_size[:, 1] / 1000,
                    meta_trajectory_size[:, 2] / 1000,
                    label="Dyad" if size == 2 else "Individual",
                    c=curvature,
                    marker=m,
                    s=10,
                    cmap=cmap_curvature,
                    vmin=0,
                    vmax=1,
                    alpha=0.5,
                )

                sc_velocity = ax[1].scatter(
                    meta_trajectory_size[:, 1] / 1000,
                    meta_trajectory_size[:, 2] / 1000,
                    label="Dyad" if size == 2 else "Individual",
                    c=velocity,
                    marker=m,
                    s=10,
                    cmap=cmap_velocity,
                    vmin=0.5,
                    vmax=1.5,
                    alpha=0.8,
                )

            divider_curvature = make_axes_locatable(ax[0])
            cax_curvature = divider_curvature.append_axes("right", size="2%", pad=0.05)
            fig.colorbar(sc_curvature, cax=cax_curvature, label="$\\kappa$ [1/m]")
            divider_velocity = make_axes_locatable(ax[1])
            cax_velocity = divider_velocity.append_axes("right", size="2%", pad=0.05)
            fig.colorbar(sc_velocity, cax=cax_velocity, label="$v$ [m/s]")

            curvature_legend_color = cmap_curvature(0.5)
            velocity_legend_color = cmap_velocity(0.5)

            custom_legend = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=curvature_legend_color,
                    markersize=8,
                    label="Individual",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor=velocity_legend_color,
                    markersize=8,
                    label="Dyad",
                ),
            ]
            ax[0].legend(handles=custom_legend, loc="upper right")
            ax[1].legend(handles=custom_legend, loc="upper right")

            ax[0].set_xlabel("x [m]")
            ax[0].set_ylabel("y [m]")
            ax[0].set_aspect("equal")
            ax[0].set_title("(a)")

            ax[1].set_xlabel("x [m]")
            ax[1].set_ylabel("y [m]")
            ax[1].set_aspect("equal")
            ax[1].set_title("(b)")

            plt.tight_layout()
            plt.savefig(
                f"../data/figures/02_08_trajectories_with_velocity_curvature/{day}_{source}_{sink}_trajectory.pdf"
            )
            plt.close()

    for size in [1, 2, "all"]:
        velocities[size] = np.array(velocities[size])
        curvatures[size] = np.array(curvatures[size])
        print(size, np.mean(velocities[size]), np.mean(curvatures[size]))
    indexes = np.array(indexes)

    set_indexes = set(indexes)

    # plot velocity vs curvature (scatter)

    # create random colors
    colors = ["blue", "orange", "green", "orange", "purple", "brown", "pink"]
    markers = ["o", "s", "^", "v", "D"]

    colors_markers = [(c, m) for c in colors for m in markers]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, idx in enumerate(sorted(set_indexes)):
        mask = indexes == idx

        ax.scatter(
            curvatures[2][mask],
            velocities[2][mask],
            alpha=0.3,
            s=5,
            label=idx,
            color=colors_markers[i][0],
            marker=colors_markers[i][1],
        )

    bin_centers, bin_edges, means, stds, errors, _, _, _, n_values = (
        compute_binned_values(
            curvatures[2], velocities[2], n_bins=N_BINS, equal_frequency=True
        )
    )

    ax.plot(bin_centers, means, color="black", label="Binned values (all)")
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

    # show bin edges
    for edge in bin_edges:
        ax.axvline(edge, color="gray", linestyle="-", linewidth=1)

    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("$\\kappa$ [1/m]")
    ax.set_ylabel("$v$ [m/s]")
    ax.legend()
    plt.savefig("../data/figures/02_08_velocity_curvature.pdf")
    plt.close()

    # # plot delta_velocity vs curvature (scatter)
    # fig, ax = plt.subplots(figsize=(12, 6))

    # d_vel = np.array(velocities[2]) - np.array(velocities[1])

    # ax.scatter(curvatures["all"], d_vel, alpha=0.3, s=3)

    # # fit a line
    # m, b = np.polyfit(curvatures["all"], d_vel, 1)
    # ax.plot(
    #     curvatures["all"],
    #     m * np.array(curvatures["all"]) + b,
    #     color="black",
    #     label=f"y = {m:.2f}x + {b:.2f}",
    # )

    # ax.set_xlabel("$\\kappa$ [1/m]")
    # ax.set_ylabel("$\\Delta v$ (Size 2 - Size 1) [m/s]")
    # ax.legend()
    # plt.savefig("../data/figures/02_08_delta_velocity_curvature.pdf")
    # plt.close()

    # # plot binned delta_velocity vs curvature
    # fig, ax = plt.subplots(figsize=(12, 6))

    # bin_centers, _, means, stds, errors, n_values = compute_binned_values(
    #     curvatures["all"], d_vel, n_bins=N_BINS
    # )

    # bin_centers_eq, _, means_eq, stds_eq, errors_eq, n_value_eqs = (
    #     compute_binned_values(
    #         curvatures["all"], d_vel, n_bins=N_BINS, equal_frequency=True
    #     )
    # )

    # ax.plot(bin_centers, means, color="blue", label="Bin centers")
    # ax.fill_between(
    #     bin_centers,
    #     means - stds,
    #     means + stds,
    #     color="blue",
    #     alpha=0.3,
    # )
    # ax.errorbar(
    #     bin_centers,
    #     means,
    #     yerr=errors,
    #     fmt="o",
    #     color="blue",
    #     capsize=2,
    # )

    # ax.plot(bin_centers_eq, means_eq, color="orange", label="Equal frequency")
    # ax.fill_between(
    #     bin_centers_eq,
    #     means_eq - stds_eq,
    #     means_eq + stds_eq,
    #     color="orange",
    #     alpha=0.3,
    # )
    # ax.errorbar(
    #     bin_centers_eq,
    #     means_eq,
    #     yerr=errors_eq,
    #     fmt="o",
    #     color="orange",
    #     capsize=2,
    # )

    # ax.set_xlabel("$\\kappa$ [1/m]")
    # ax.set_ylabel("$\\Delta v$ (Size 2 - Size 1) [m/s]")
    # ax.legend()

    # plt.savefig("../data/figures/02_08_delta_velocity_curvature_binned.pdf")
    # plt.close()

    # plot binned velocity vs curvature for both sizes

    fig, ax = plt.subplots(figsize=(12, 6))

    for size, color in zip([1, 2], ["orange", "blue"]):

        bin_centers, _, means, stds, errors, _, _, _, n_values = compute_binned_values(
            curvatures[size], velocities[size], n_bins=N_BINS, equal_frequency=True
        )

        ax.plot(
            bin_centers,
            means,
            color=color,
            label="Dyad" if size == 2 else "Individual",
        )
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

    plt.savefig("../data/figures/02_08_velocity_curvature_binned_velocity.pdf")
    plt.close()

    # plot binned log velocity vs log curvature for both sizes

    fig, ax = plt.subplots(figsize=(10, 5))

    for size, color in zip([1, 2], ["orange", "blue"]):

        bin_centers, _, means, stds, errors, _, _, _, n_values = compute_binned_values(
            np.log(curvatures[size]),
            np.log(velocities[size]),
            # velocities[size],
            n_bins=N_BINS,
            # equal_frequency=True,
        )

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
            fmt="-o",
            color=color,
            capsize=2,
            label="Dyad" if size == 2 else "Individual",
        )

    ax.set_xlabel("$\\log(\\kappa)$")
    ax.set_ylabel("$\\log(v)$")

    ax.legend()
    ax.grid(color="gray", linestyle="--", linewidth=0.5)

    plt.savefig("../data/figures/02_08_velocity_curvature_binned_log_velocity.pdf")

    # plot binned log velocity vs log curvature for both sizes (with hinge)

    fig, ax = plt.subplots(figsize=(10, 5))

    all_hinge_params = []

    for size, color in zip([1, 2], ["orange", "blue"]):

        bin_centers, _, means, stds, errors, _, _, _, n_values = compute_binned_values(
            np.log(curvatures[size]),
            np.log(velocities[size]),
            # velocities[size],
            n_bins=N_BINS,
            # equal_frequency=True,
        )

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
            fmt="-o",
            color=color,
            capsize=2,
            label="Dyad" if size == 2 else "Individual",
        )

        hinge_params, r_squared = fit_hinge_function(bin_centers, means)
        all_hinge_params.append(hinge_params)
        plot_hinge_function(
            np.min(bin_centers),
            np.max(bin_centers),
            *hinge_params,
            ax,
            label=f"$\\log(v_h) = {hinge_params[1]:.2f}, \\log(\\kappa_h) = {hinge_params[0]:.2f}, \\gamma = {hinge_params[2]:.2f}$ (R$^2$ = {r_squared:.2f})",
            color=color,
        )

    ax.set_xlabel("$\\log(\\kappa)$")
    ax.set_ylabel("$\\log(v)$")

    ax.legend()
    ax.grid(color="gray", linestyle="--", linewidth=0.5)

    plt.savefig(
        "../data/figures/02_08_velocity_curvature_binned_log_velocity_with_hinge.pdf"
    )

    #  make a table with hinge parameters
    with open("../data/tables/02_08_hinge_parameters_velocity_curvature.tex", "w") as f:
        f.write("\\begin{table}\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Condition & $\\log(\\kappa_h)$ & $\\log(v_h)$ & $\\gamma$ \\\\\n")
        f.write("\\midrule\n")
        for size, hinge_params in zip([1, 2], all_hinge_params):
            f.write(
                f" {'Dyad' if size == 2 else 'Individual'} & {hinge_params[0]:.2f} ({np.exp(hinge_params[0]):.2f}) & {hinge_params[1]:.2f} ({np.exp(hinge_params[1]):.2f}) & {hinge_params[2]:.2f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Hinge parameters for velocity-curvature relationship.}\n")
        f.write("\\label{tab:velocity_curvature_hinge_parameters}\n")
        f.write("\\end{table}\n")

    # plot binned log velocity vs log radius for both sizes

    fig, ax = plt.subplots(figsize=(12, 6))

    for size, color in zip([1, 2], ["orange", "blue"]):

        bin_centers, _, means, stds, errors, _, _, _, n_values = compute_binned_values(
            np.log(1 / curvatures[size]),
            np.log(velocities[size]),
            # velocities[size],
            n_bins=N_BINS,
            equal_frequency=True,
        )

        ax.plot(
            bin_centers,
            means,
            color=color,
            label="Dyad" if size == 2 else "Individual",
        )
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
        "../data/figures/02_08_velocity_curvature_binned_log_velocity_radius.pdf"
    )
