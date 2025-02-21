import numpy as np
import matplotlib.pyplot as plt
from utils import (
    compute_curvature,
    compute_angular_velocity,
    compute_space_bin_average_trajectory,
)
from parameters import SIZE_BIN

import scienceplots

plt.style.use("science")


def make_trajectory_turn(v, dt):
    p_start = np.array([0, 0])
    p = p_start.copy()
    positions = [p.copy()]
    times = [0]
    # straight up
    y_start_turn = 5000
    while p[1] < y_start_turn:
        p[1] += v * dt
        positions.append(p.copy())
        times.append(times[-1] + dt)

    # turn right
    R = 1000  # radius, mm
    w = v / R  # angular velocity, rad/s
    T = (np.pi / 2) * R / v  # duration of quarter-circle
    center = np.array([p[0] + R, p[1]])
    theta = np.pi
    n_steps = int(T / dt)

    for _ in range(n_steps):
        theta -= w * dt  # Decrease angle (turn right)
        p[0] = center[0] + R * np.cos(theta)
        p[1] = center[1] + R * np.sin(theta)
        positions.append(p.copy())
        times.append(times[-1] + dt)

    # straight right
    x_end = 6000
    while p[0] < x_end:
        p[0] += v * dt
        positions.append(p.copy())
        times.append(times[-1] + dt)

    positions = np.array(positions)

    trajectory = np.zeros((len(positions), 7))
    trajectory[:, 0] = times
    trajectory[:, 1] = positions[:, 0]
    trajectory[:, 2] = positions[:, 1]

    return trajectory


def make_trajectory_circular(v, R, dt=0.01):
    omega = v / R
    n_turns = 2
    T = 2 * np.pi * R / v * n_turns
    n_steps = int(T / dt)

    t = np.linspace(0, T, n_steps)
    x = R * np.cos(omega * t)
    y = R * np.sin(omega * t)

    trajectory = np.zeros((len(t), 7))
    trajectory[:, 0] = t
    trajectory[:, 1] = x
    trajectory[:, 2] = y

    return trajectory


if __name__ == "__main__":

    v = 2000  # velocity, mm/s
    R = 5000  # radius, mm

    for dt in [0.01, 0.05, 0.1, 0.5, 1]:
        # trajectory = make_trajectory_turn(v, dt)
        trajectory = make_trajectory_circular(v, R, dt)

        # manhattan distance
        # distance = np.abs(trajectory[0, 1] - trajectory[-1, 1]) + np.abs(
        #     trajectory[0, 2] - trajectory[-1, 2]
        # )
        # n_points_trajectory = int(np.floor(distance / SIZE_BIN))

        # space_bin_average_trajectory = compute_space_bin_average_trajectory(
        #     trajectory, n_points_trajectory, time="average"
        # )

        vel_mag, curvature = compute_curvature(trajectory, 10)
        curvature = np.abs(curvature)
        radius = 1 / curvature

        error_radius = np.abs(radius - R)

        mean_error_radius = np.mean(error_radius)

        angular_velocity_from_curvature = vel_mag / radius
        omega = np.abs(compute_angular_velocity(trajectory, None))

        fig, ax = plt.subplots(2, 3, figsize=(12, 5))

        ax[0][0].scatter(
            trajectory[:, 1] / 1000,
            trajectory[:, 2] / 1000,
            s=3,
            alpha=np.linspace(0.1, 1, len(trajectory)),
        )
        ax[0][0].set_xlabel("$x$ [m]")
        ax[0][0].set_ylabel("$y$ [m]")
        ax[0][0].axis("equal")

        ax[1][0].plot(trajectory[:, 0], vel_mag / 1000)
        ax[1][0].set_xlabel("$t$ [s]")
        ax[1][0].set_ylabel("$v$ [m/s]")
        ax[1][0].set_title("Velocity Magnitude")
        ax[1][0].set_ylim([0, 5])
        ax[1][0].hlines(v / 1000, 0, trajectory[-1, 0], "r", "--")
        ax[1][0].grid(color="gray", linestyle="--", linewidth=0.5)

        ax[0][1].plot(trajectory[:, 0], radius / 1000)
        ax[0][1].set_xlabel("$t$ [s]")
        ax[0][1].set_ylabel("$R$ [mm]")
        ax[0][1].set_title("Radius of Curvature")
        ax[0][1].set_ylim([0, 10])
        ax[0][1].hlines(R / 1000, 0, trajectory[-1, 0], "r", "--")
        ax[0][1].grid(color="gray", linestyle="--", linewidth=0.5)

        ax[1][1].plot(trajectory[:, 0], angular_velocity_from_curvature)
        ax[1][1].plot(trajectory[:, 0], omega)
        ax[1][1].hlines(v / R, 0, trajectory[-1, 0], "r", "--")
        ax[1][1].set_ylim([0, 1])
        ax[1][1].set_xlabel("$t$ [s]")
        ax[1][1].set_ylabel("$\omega$ [rad/s]")
        ax[1][1].set_title("Angular Velocity")
        ax[1][1].legend(["From Curvature", "From Trajectory"])
        ax[1][1].grid(color="gray", linestyle="--", linewidth=0.5)

        ax[0][2].plot(trajectory[:, 0], error_radius / 1000)
        ax[0][2].set_xlabel("$t$ [s]")
        ax[0][2].set_ylabel("$\Delta R$ [mm]")
        ax[0][2].set_title("Error in Radius of Curvature")
        ax[0][2].text(
            0.5,
            0.5,
            f"Mean Error: {mean_error_radius/1000:.4f} m",
            transform=ax[0][2].transAxes,
        )
        ax[0][2].grid(color="gray", linestyle="--", linewidth=0.5)

        plt.tight_layout()
        # plt.show()
        plt.savefig(
            f"../data/figures/test_angular_velocity/angular_velocity_dt_{dt}.pdf"
        )

    # plt.tight_layout()
    # plt.show()
