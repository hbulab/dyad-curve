import numpy as np
from utils import compute_space_bin_average_trajectory
import matplotlib.pyplot as plt

import scienceplots

plt.style.use("science")

if __name__ == "__main__":

    a = 0.5  # acceleration, m/s^2
    t = np.linspace(0, 10, 200)

    x = 0.5 * a * t**2
    y = np.zeros_like(x)

    vx = np.diff(x) / np.diff(t)
    vx = np.append(vx, vx[-1])
    vy = np.diff(y) / np.diff(t)
    vy = np.append(vy, vy[-1])

    ax = np.diff(vx) / np.diff(t)
    ax = np.append(ax, ax[-1])
    ay = np.diff(vy) / np.diff(t)
    ay = np.append(ay, ay[-1])

    trajectory = np.zeros((len(t), 7))
    trajectory[:, 0] = t
    trajectory[:, 1] = x
    trajectory[:, 2] = y
    trajectory[:, 5] = vx
    trajectory[:, 6] = vy

    v_mag = np.sqrt(vx**2 + vy**2)
    a_mag = np.sqrt(ax**2 + ay**2)

    fig, axes = plt.subplots(4, 1, figsize=(7, 7))

    for method in ["interp", "index", "average"]:

        space_bin_average_trajectory = compute_space_bin_average_trajectory(
            trajectory, 10, time=method
        )

        v_binned_mag = np.sqrt(
            space_bin_average_trajectory[:, 5] ** 2
            + space_bin_average_trajectory[:, 6] ** 2
        )
        ax_binned = np.diff(space_bin_average_trajectory[:, 5]) / np.diff(
            space_bin_average_trajectory[:, 0]
        )
        ax_binned = np.append(ax_binned, ax_binned[-1])
        ay_binned = np.diff(space_bin_average_trajectory[:, 6]) / np.diff(
            space_bin_average_trajectory[:, 0]
        )
        ay_binned = np.append(ay_binned, ay_binned[-1])
        a_binned_mag = np.sqrt(ax_binned**2 + ay_binned**2)

        axes[1].plot(
            space_bin_average_trajectory[:, 0],
            space_bin_average_trajectory[:, 1],
            "-o",
            label=method,
            markersize=5,
            alpha=0.5,
        )
        axes[2].plot(
            space_bin_average_trajectory[:, 0],
            v_binned_mag,
            "-o",
            label=method,
            markersize=5,
            alpha=0.5,
        )
        axes[3].plot(
            space_bin_average_trajectory[:, 0],
            a_binned_mag,
            "-o",
            label=method,
            markersize=5,
            alpha=0.5,
        )

    axes[0].scatter(x, y, label="Original", s=3)
    axes[0].scatter(
        space_bin_average_trajectory[:, 1],
        space_bin_average_trajectory[:, 2],
        label="Binned",
        s=10,
    )
    axes[0].set_xlabel("$x$ [m]")
    axes[0].set_ylabel("$y$ [m]")
    axes[0].axis("equal")
    axes[0].grid(color="gray", linestyle="--", linewidth=0.5)
    axes[0].legend()

    axes[1].plot(t, x, label="Original")

    axes[1].set_ylabel("$x$ [m]")
    axes[1].set_xlabel("$t$ [s]")
    axes[1].grid(color="gray", linestyle="--", linewidth=0.5)
    axes[1].legend()

    axes[2].plot(t, v_mag, label="Original")

    axes[2].set_ylabel("$v$ [m/s]")
    axes[2].set_xlabel("$t$ [s]")
    axes[2].grid(color="gray", linestyle="--", linewidth=0.5)
    axes[2].legend()

    axes[3].plot(t, a_mag, label="Original")

    axes[3].hlines(a, 0, 10, "k", "--", label="True Acceleration")
    axes[3].set_ylabel("$a$ [m/s$^2$]")
    axes[3].set_xlabel("$t$ [s]")
    axes[3].grid(color="gray", linestyle="--", linewidth=0.5)
    axes[3].legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/figures/test_binned_time.pdf")
    plt.close()
