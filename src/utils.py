import pickle as pk
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


def pickle_load(file_path: str):
    """Load the content of a pickle file

    Parameters
    ----------
    file_path : str
        The path to the file which will be unpickled

    Returns
    -------
    obj
        The content of the pickle file
    """
    with open(file_path, "rb") as f:
        data = pk.load(f)
    return data


def pickle_save(data, file_path: str):
    """Save data to pickle file

    Parameters
    ----------
    file_path : str
        The path to the file where the data will be saved
    data : obj
        The data to save
    """
    with open(file_path, "wb") as f:
        pk.dump(data, f)


def compute_simultaneous_observations(trajectories: list[np.ndarray]) -> list:
    """Find the section of the trajectories that correspond to simultaneous observations

    Parameters
    ----------
    trajectories : list
        List of trajectories

    Returns
    -------
    list
        The list of trajectories with simultaneous observations (i.e. same time stamps)
    """

    simult_time = trajectories[0][:, 0]

    for trajectory in trajectories[1:]:
        simult_time = np.intersect1d(simult_time, trajectory[:, 0])

    simult_trajectories = []
    for trajectory in trajectories:
        time_mask = np.isin(trajectory[:, 0], simult_time)
        simult_trajectory = trajectory[time_mask, :]
        simult_trajectories += [simult_trajectory]

    return simult_trajectories


def compute_center_of_mass_trajectory(trajectories: list[np.ndarray]) -> np.ndarray:
    """Computes the center of mass of a list of trajectories. Position and velocities are
    the average of all trajectories.

    Parameters
    ----------
    trajectories : list[np.ndarray]
        A list of trajectories

    Returns
    -------
    np.ndarray
        The trajectory of the center of mass
    """
    simultaneous_traj = compute_simultaneous_observations(trajectories)
    n_traj = len(trajectories)

    simultaneous_time = simultaneous_traj[0][:, 0]
    x_members = np.stack([traj[:, 1] for traj in simultaneous_traj], axis=1)
    y_members = np.stack([traj[:, 2] for traj in simultaneous_traj], axis=1)
    z_members = np.stack([traj[:, 3] for traj in simultaneous_traj], axis=1)

    vx_members = np.stack([traj[:, 5] for traj in simultaneous_traj], axis=1)
    vy_members = np.stack([traj[:, 6] for traj in simultaneous_traj], axis=1)

    x_center_of_mass = np.sum(x_members, axis=1) / n_traj
    y_center_of_mass = np.sum(y_members, axis=1) / n_traj
    z_center_of_mass = np.sum(z_members, axis=1) / n_traj

    vx_center_of_mass = np.sum(vx_members, axis=1) / n_traj
    vy_center_of_mass = np.sum(vy_members, axis=1) / n_traj

    v_center_of_mass = (vx_center_of_mass**2 + vx_center_of_mass**2) ** 0.5

    trajectory = np.stack(
        (
            simultaneous_time,
            x_center_of_mass,
            y_center_of_mass,
            z_center_of_mass,
            v_center_of_mass,
            vx_center_of_mass,
            vy_center_of_mass,
        ),
        axis=1,
    )
    return trajectory


def filter_groups_by_size(groups, size):
    """Filter groups by size

    Parameters
    ----------
    groups : list
        List of groups
    size : int
        Size of the group

    Returns
    -------
    list
        List of groups with the desired size
    """
    return [group for group in groups if len(group["members"]) == size]


def filter_groups_by_day(groups, day):
    """Filter groups by day

    Parameters
    ----------
    groups : list
        List of groups
    day : str
        Day

    Returns
    -------
    list
        List of groups from the desired day
    """
    return [group for group in groups if group["day"] == day]


def filter_groups_by_social_binding(groups, soc_binding):
    """Filter groups by social binding

    Parameters
    ----------
    groups : list
        List of groups
    soc_binding : bool
        Social binding

    Returns
    -------
    list
        List of groups with the desired social binding
    """
    return [group for group in groups if group["soc_binding"] == soc_binding]


def interpolate_trajectory(trajectory, n_points):
    """Interpolate a trajectory

    Parameters
    ----------
    trajectory : np.ndarray
        The trajectory to interpolate
    n_points : int
        The number of points to interpolate

    Returns
    -------
    np.ndarray
        The interpolated trajectory
    """
    x = trajectory[:, 1]
    y = trajectory[:, 2]

    vx = trajectory[:, 5]
    vy = trajectory[:, 6]

    x_interp = np.interp(np.linspace(0, 1, n_points), np.linspace(0, 1, len(x)), x)
    y_interp = np.interp(np.linspace(0, 1, n_points), np.linspace(0, 1, len(y)), y)

    vx_interp = np.interp(np.linspace(0, 1, n_points), np.linspace(0, 1, len(vx)), vx)
    vy_interp = np.interp(np.linspace(0, 1, n_points), np.linspace(0, 1, len(vy)), vy)

    t_interp = np.linspace(0, 1, n_points)

    interpolated_trajectory = np.zeros((n_points, 7))
    interpolated_trajectory[:, 0] = t_interp
    interpolated_trajectory[:, 1] = x_interp
    interpolated_trajectory[:, 2] = y_interp
    interpolated_trajectory[:, 5] = vx_interp
    interpolated_trajectory[:, 6] = vy_interp

    return interpolated_trajectory


def compute_time_bin_average_trajectory(trajectory, n_points):
    """Compute an average trajectory over n_points binned by normalized time.

    Parameters
    ----------
    trajectory : np.ndarray
        The trajectory to average
    n_points : int
        The number of points in the average trajectory

    Returns
    -------
    np.ndarray
        The average trajectory
    """
    t = trajectory[:, 0]
    t_normalized = (t - t[0]) / (t[-1] - t[0])

    bins_edges = np.linspace(0, 1, n_points + 1)
    bin_centers = (bins_edges[1:] + bins_edges[:-1]) / 2
    indices = np.digitize(t_normalized, bins_edges) - 1

    average_trajectory = np.zeros((n_points, trajectory.shape[1]))
    bin_counts = np.zeros(n_points)

    for i, bin_idx in enumerate(indices):
        if 0 <= bin_idx < n_points:
            average_trajectory[bin_idx] += trajectory[i]
            bin_counts[bin_idx] += 1

    # compute the average for each bin
    for i in range(n_points):
        if bin_counts[i] > 0:
            average_trajectory[i] /= bin_counts[i]
        else:
            average_trajectory[i] = np.nan

    average_trajectory[:, 0] = bin_centers

    nan_mask = np.isnan(average_trajectory[:, 0])
    valid_indices = np.where(~nan_mask)[0]
    invalid_indices = np.where(nan_mask)[0]

    for dim in range(1, trajectory.shape[1]):
        valid_values = average_trajectory[valid_indices, dim]
        average_trajectory[invalid_indices, dim] = np.interp(
            bin_centers[invalid_indices],
            bin_centers[valid_indices],
            valid_values,
        )

    #     fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    #     axes[0].scatter(
    #         t_normalized,
    #         trajectory[:, 1],
    #         label="Original trajectory",
    #         s=1,
    #     )
    #     axes[0].plot(
    #         bin_centers,
    #         average_trajectory[:, 1],
    #         label="Average trajectory",
    #         color="red",
    #     )
    #     axes[0].set_title("X coordinate")
    #     axes[0].set_xlabel("Normalized time")
    #     axes[0].set_ylabel("X coordinate")

    #     axes[1].scatter(
    #         t_normalized,
    #         trajectory[:, 2],
    #         label="Original trajectory",
    #         s=1,
    #     )
    #     axes[1].plot(
    #         bin_centers,
    #         average_trajectory[:, 2],
    #         label="Average trajectory",
    #         color="red",
    #     )
    #     axes[1].set_title("Y coordinate")
    #     axes[1].set_xlabel("Normalized time")
    #     axes[1].set_ylabel("Y coordinate")

    #     # vertical lines to show the bins
    #     for bin_edge in bins_edges:
    #         axes[0].axvline(bin_edge, color="gray", linewidth=0.5)
    #         axes[1].axvline(bin_edge, color="gray", linewidth=0.5)

    #     axes[0].legend()
    #     axes[1].legend()

    #     plt.show()

    return average_trajectory


def compute_space_bin_average_trajectory(trajectory, n_bins, time="interp"):
    """Compute an average trajectory over n_bins binned by space.

    Parameters
    ----------
    trajectory : np.ndarray
        The trajectory to average
    n_bins : int
        The number of bins in the average trajectory
    time : str
        The time values to use for the average trajectory

    Returns
    -------
    np.ndarray
        The average trajectory
    """
    distances = np.sqrt(np.sum(np.diff(trajectory[:, 1:3], axis=0) ** 2, axis=1))
    arc_length = np.insert(np.cumsum(distances), 0, 0)

    bins_edges = np.linspace(0, arc_length[-1], n_bins + 1)

    bin_centers = (bins_edges[1:] + bins_edges[:-1]) / 2
    indices = np.digitize(arc_length, bins_edges) - 1

    average_trajectory = np.zeros((n_bins, trajectory.shape[1]))
    bin_counts = np.zeros(n_bins)

    for i, bin_idx in enumerate(indices):
        if 0 <= bin_idx < n_bins:
            average_trajectory[bin_idx] += trajectory[i]
            bin_counts[bin_idx] += 1

    # compute the average for each bin
    for i in range(n_bins):
        if bin_counts[i] > 0:
            average_trajectory[i] /= bin_counts[i]
        else:
            average_trajectory[i] = np.nan

    nan_mask = np.isnan(average_trajectory[:, 0])
    valid_indices = np.where(~nan_mask)[0]
    invalid_indices = np.where(nan_mask)[0]

    for dim in range(trajectory.shape[1]):
        valid_values = average_trajectory[valid_indices, dim]
        average_trajectory[invalid_indices, dim] = np.interp(
            bin_centers[invalid_indices],
            bin_centers[valid_indices],
            valid_values,
        )

    if time == "index":
        average_trajectory[:, 0] = np.arange(n_bins)
    elif time == "interp":
        average_trajectory[:, 0] = np.interp(
            bin_centers,  # Arc-length of binned trajectory
            arc_length,  # Arc-length of original trajectory
            trajectory[:, 0],  # Original time values
        )

    # fig, ax = plt.subplots(figsize=(8, 6))

    # ax.scatter(
    #     trajectory[:, 1],
    #     trajectory[:, 2],
    #     label="Original trajectory",
    #     s=1,
    # )
    # ax.plot(
    #     average_trajectory[:, 1],
    #     average_trajectory[:, 2],
    #     label="Average trajectory",
    #     color="red",
    # )

    # ax.set_title("Average trajectory")
    # ax.set_xlabel("X coordinate")
    # ax.set_ylabel("Y coordinate")
    # ax.legend()
    # ax.set_aspect("equal")

    # plt.show()

    # # show velocity
    # fig, ax = plt.subplots(figsize=(8, 6))

    # vel_mag = np.sqrt(trajectory[:, 5] ** 2 + trajectory[:, 6] ** 2)
    # vel_mag_avg = np.sqrt(
    #     average_trajectory[:, 5] ** 2 + average_trajectory[:, 6] ** 2
    # )

    # ax.plot(
    #     trajectory[:, 0],
    #     vel_mag,
    #     label="Original trajectory",
    #     color="blue",
    # )
    # ax.plot(
    #     average_trajectory[:, 0],
    #     vel_mag_avg,
    #     label="Average trajectory",
    #     color="red",
    # )

    # ax.set_title("Velocity magnitude")
    # ax.set_xlabel("Time")
    # ax.set_ylabel("Velocity magnitude")
    # ax.legend()

    # plt.show()

    # # show acceleration
    # fig, ax = plt.subplots(figsize=(8, 6))

    # acc_mag = np.sqrt(trajectory[:, 7] ** 2 + trajectory[:, 8] ** 2)
    # acc_mag_avg = np.sqrt(
    #     average_trajectory[:, 7] ** 2 + average_trajectory[:, 8] ** 2
    # )

    # ax.plot(
    #     trajectory[:, 0],
    #     acc_mag,
    #     label="Original trajectory",
    #     color="blue",
    # )

    # ax.plot(
    #     average_trajectory[:, 0],
    #     acc_mag_avg,
    #     label="Average trajectory",
    #     color="red",
    # )

    # ax.set_title("Acceleration magnitude")
    # ax.set_xlabel("Time")
    # ax.set_ylabel("Acceleration magnitude")
    # ax.legend()

    # plt.show()

    return average_trajectory


def compute_mask_in_source(trajectory, source):
    """Check if a trajectory is inside a source

    Parameters
    ----------
    trajectory : np.ndarray
        The trajectory to check
    source : dict
        The source

    Returns
    -------
    np.ndarray
        A boolean mask indicating if the trajectory is inside the source
    """
    return (
        (trajectory[:, 1] >= source["xmin"])
        & (trajectory[:, 1] <= source["xmax"])
        & (trajectory[:, 2] >= source["ymin"])
        & (trajectory[:, 2] <= source["ymax"])
    )


def compute_trajectory_source_to_sink(trajectory, source, sink):
    """Compute the trajectory from a source to a sink.
    The trajectory will start when the trajectory leaves the source and end when it enters the sink.

    Parameters
    ----------
    trajectory : np.ndarray
        The trajectory
    source : dict
        The source
    sink : dict
        The sink

    Returns
    -------
    np.ndarray
        The trajectory from source to sink
    """
    mask_in_source = compute_mask_in_source(trajectory, source)
    mask_in_sink = compute_mask_in_source(trajectory, sink)

    if not np.any(mask_in_source) or not np.any(mask_in_sink):
        return None

    tmin = trajectory[:, 0][mask_in_source][-1]
    tmax = trajectory[:, 0][mask_in_sink][0]

    if tmin >= tmax:
        return None

    trajectory_source_to_sink = trajectory[
        (trajectory[:, 0] >= tmin) & (trajectory[:, 0] <= tmax)
    ]

    return trajectory_source_to_sink


def compute_binned_values(
    x_values, y_values, n_bins, min_v=None, max_v=None, equal_frequency=False
) -> tuple:
    """Compute binned values of y_values with respect to x_values

    Parameters
    ----------
    x_values : np.ndarray
        The x values
    y_values : np.ndarray
        The y values
    min_v : float
        The minimum value of the bins
    max_v : float
        The maximum value of the bins
    n_bins : int
        The number of bins
    equal_frequency : bool
        If True, the bins will have equal frequency

    Returns
    -------
    tuple
        bin_centers : np.ndarray
            The centers of the bins
        bin_edges: np.ndarray
            The edges of the bins
        means : np.ndarray
            The mean of the y values in each bin
        stds : np.ndarray
            The standard deviation of the y values in each bin
        errors : np.ndarray
            The standard error of the y values in each bin
        n_values : np.ndarray
            The number of values
    """

    if min_v is None:
        min_v = np.nanmin(x_values)
    if max_v is None:
        max_v = np.nanmax(x_values)

    if not equal_frequency:
        pdf_edges = np.linspace(min_v, max_v, n_bins + 1)
    else:
        pdf_edges = np.interp(
            np.linspace(0, 1, n_bins + 1),
            np.linspace(0, 1, len(x_values)),
            np.sort(x_values),
        )

    bin_centers = 0.5 * (pdf_edges[0:-1] + pdf_edges[1:])
    indices = np.digitize(x_values, pdf_edges) - 1

    means = np.full(n_bins, np.nan)
    stds = np.full(n_bins, np.nan)
    errors = np.full(n_bins, np.nan)
    n_values = np.zeros(n_bins)

    for i in range(n_bins):
        if np.sum(indices == i) == 0:
            continue
        means[i] = np.nanmean(y_values[indices == i])
        stds[i] = np.nanstd(y_values[indices == i])
        errors[i] = stds[i] / np.sqrt(np.sum(indices == i))
        n_values[i] = np.sum(indices == i)

    return bin_centers, pdf_edges, means, stds, errors, n_values


def transform_trajectories_to_reference_frame(
    trajectory_ref: np.ndarray,
    trajectories: list[np.ndarray],
    axis: str = "x",
    nullify_velocities: bool = True,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Transform the trajectories in such a way that position for trajectory_ref will always
     be at the origin and velocity for A will be aligned along the positive x axis.
     Other trajectories are moved to the same reference frame

    Parameters
    ----------
    trajectory_ref : np.ndarray
        The trajectory that will be aligned with the x axis
    trajectories : list[np.ndarray]
        A list of trajectories to transform in that reference frame
    axis : str
        The axis along which the velocity will be aligned
    nullify_velocities : bool
        If True, the velocities will be nullified

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The aligned trajectories
    """
    simultaneous_trajectories = compute_simultaneous_observations(
        [trajectory_ref] + trajectories
    )
    trajectory_ref = simultaneous_trajectories[0]
    trajectories = simultaneous_trajectories[1:]

    # build the rotation matrix
    rotation_matrices = np.zeros(
        (len(trajectory_ref), 2, 2)
    )  # 1 rotation matrix per observation

    vel_mag = np.linalg.norm(trajectory_ref[:, 5:7], axis=1)
    vel_mag_mask = vel_mag > 0
    cos_rot = np.full(len(trajectory_ref), np.nan)
    sin_rot = np.full(len(trajectory_ref), np.nan)

    if axis == "x":
        cos_rot[vel_mag_mask] = trajectory_ref[vel_mag_mask, 5] / vel_mag[vel_mag_mask]
        sin_rot[vel_mag_mask] = -trajectory_ref[vel_mag_mask, 6] / vel_mag[vel_mag_mask]
    elif axis == "y":
        cos_rot[vel_mag_mask] = trajectory_ref[vel_mag_mask, 6] / vel_mag[vel_mag_mask]
        sin_rot[vel_mag_mask] = trajectory_ref[vel_mag_mask, 5] / vel_mag[vel_mag_mask]

    rotation_matrices[:, 0, 0] = cos_rot
    rotation_matrices[:, 0, 1] = -sin_rot
    rotation_matrices[:, 1, 0] = sin_rot
    rotation_matrices[:, 1, 1] = cos_rot

    transformed_ref = trajectory_ref.copy()
    # translate the position to have it always at 0, 0
    transformed_ref[:, 1:3] -= trajectory_ref[:, 1:3]
    # translate the velocities
    if nullify_velocities:
        transformed_ref[:, 5:7] -= trajectory_ref[:, 5:7]
    # rotate the reference velocity
    transformed_ref[:, 5:7] = np.diagonal(
        np.dot(rotation_matrices, transformed_ref[:, 5:7].T), axis1=0, axis2=2
    ).T

    transformed_trajectories = []
    for trajectory in trajectories:
        transformed_trajectory = trajectory.copy()
        # transform the trajectory
        pos = transformed_trajectory[:, 1:3]
        pos -= trajectory_ref[:, 1:3]
        rotated_pos = np.diagonal(np.dot(rotation_matrices, pos.T), axis1=0, axis2=2).T
        transformed_trajectory[:, 1:3] = rotated_pos

        # transform the velocities
        vel = transformed_trajectory[:, 5:7]
        if nullify_velocities:
            vel -= trajectory_ref[:, 5:7]
        rotated_vel = np.diagonal(np.dot(rotation_matrices, vel.T), axis1=0, axis2=2).T
        transformed_trajectory[:, 5:7] = rotated_vel
        transformed_trajectories += [transformed_trajectory]

    return transformed_ref, transformed_trajectories


def filter_bad_trajectories(
    trajectories, cell_size, min_n_trajectories_bad, ratio_bad_cells
):
    """Filter out the bad trajectories

    Parameters
    ----------
    trajectories : np.ndarray
        The list of trajectories
    cell_size : float
        The size of the cells for the spatial binning
    min_n_trajectories_bad : int
       A cell with less than this number of trajectories will be considered as "bad"
    ratio_bad_cells : float
        The ratio of bad cells in a trajectory to consider it as "bad"

    Returns
    -------
    np.ndarray
        The filtered trajectories
    """
    min_x = np.nanmin(trajectories[:, :, 1])
    max_x = np.nanmax(trajectories[:, :, 1])
    min_y = np.nanmin(trajectories[:, :, 2])
    max_y = np.nanmax(trajectories[:, :, 2])

    n_bin_x = int(np.ceil((max_x - min_x) / cell_size) + 1)
    n_bin_y = int(np.ceil((max_y - min_y) / cell_size) + 1)
    grid = np.zeros((n_bin_x, n_bin_y, trajectories.shape[0]))

    for i, trajectory in enumerate(trajectories):
        x = trajectory[:, 1]
        y = trajectory[:, 2]

        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]

        nx = (x - min_x) / cell_size
        ny = (y - min_y) / cell_size

        nx = np.ceil(nx).astype("int")
        ny = np.ceil(ny).astype("int")

        in_limit = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)

        nx = nx[in_limit]
        ny = ny[in_limit]

        grid[nx, ny, i] = 1

    # # show grid
    # fig, ax = plt.subplots(figsize=(12, 6))
    # ax.imshow(np.sum(grid, axis=2).T, origin="lower", cmap="viridis")
    # plt.show()

    # find grid cells with few trajectories
    grid_n_trajectories = np.sum(grid, axis=2)
    mask_grid_cells_few_trajectories = (grid_n_trajectories > 0) & (
        grid_n_trajectories <= min_n_trajectories_bad
    )

    good_trajectories = []
    for i, trajectory in enumerate(trajectories):
        count_bad_cells = np.sum(grid[mask_grid_cells_few_trajectories, i])
        total_cells = np.sum(grid[:, :, i])
        ratio_bad = count_bad_cells / total_cells
        # print(ratio_bad)

        if ratio_bad < ratio_bad_cells:
            good_trajectories.append(trajectory)

    good_trajectories = np.array(good_trajectories)

    return good_trajectories


def compute_distance(source, sink):
    """Compute the distance between two points

    Parameters
    ----------
    source : dict
        The source
    sink : dict
        The sink

    Returns
    -------
    float
        The distance between the source and the sink
    """
    center_x_source = (source["xmin"] + source["xmax"]) / 2
    center_y_source = (source["ymin"] + source["ymax"]) / 2
    center_x_sink = (sink["xmin"] + sink["xmax"]) / 2
    center_y_sink = (sink["ymin"] + sink["ymax"]) / 2

    dx = abs(center_x_sink - center_x_source)
    dy = abs(center_y_sink - center_y_source)

    distance = dx + dy  # Manhattan distance

    return distance


def compute_curvature_old(trajectory, window_size) -> tuple[np.ndarray, np.ndarray]:
    """Compute the curvature of a trajectory.

    Parameters
    ----------
    trajectory : np.ndarray
        Trajectory with shape (n_points, 2).
    window_size : int
        The size of the window for smoothing the velocity and acceleration.

    Returns
    -------
    vel_mag_smoothed : np.ndarray
        Smoothed velocity magnitude.
    curvature : np.ndarray
        Curvature of the trajectory.
    """
    # t = trajectory[:, 0] - trajectory[0, 0]
    # dt = np.mean(np.diff(t))

    # window_duration = 3  # s
    # window_size = int(window_duration / dt)

    # if window_size < 3:
    #     window_size = 3

    # velocity = trajectory[:, 5:7] / 1000
    # velocity_smoothed = savgol_filter(velocity, window_size, 2, axis=0)
    # vel_mag_smoothed = np.linalg.norm(velocity_smoothed, axis=1)

    velocity = np.gradient(trajectory[:, 1:3], trajectory[:, 0], axis=0) / 1000
    velocity_smoothed = savgol_filter(velocity, window_size, 2, axis=0)
    vel_mag_smoothed = np.linalg.norm(velocity_smoothed, axis=1)

    # acceleration = np.gradient(velocity, axis=0)
    # acceleration_smoothed = savgol_filter(acceleration, window_size, 2, axis=0)

    acceleration = np.gradient(velocity_smoothed, trajectory[:, 0], axis=0)
    acceleration_smoothed = savgol_filter(acceleration, window_size, 2, axis=0)

    # acceleration = trajectory[:, 7:9] / 1000
    # acceleration_smoothed = savgol_filter(acceleration, window_size, 2, axis=0)
    # acc_mag_smoothed = np.linalg.norm(acceleration_smoothed, axis=1)

    # fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # ax[0].plot(trajectory[:, 0], vel_mag_smoothed)
    # ax[0].set_ylabel("Velocity magnitude")

    # ax[1].plot(trajectory[:, 0], acc_mag_smoothed)
    # ax[1].set_ylabel("Acceleration magnitude")

    # plt.show()

    ax = acceleration_smoothed[:, 0]
    ay = acceleration_smoothed[:, 1]

    vx = velocity_smoothed[:, 0]
    vy = velocity_smoothed[:, 1]

    curvature = (vx * ay - vy * ax) / (vx**2 + vy**2) ** 1.5

    # fig, ax = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

    # ax[0].plot(t, vel_mag)
    # ax[0].plot(t, vel_mag_smoothed)
    # ax[0].set_ylabel("Velocity magnitude")

    # ax[1].plot(t, acc_mag)
    # ax[1].plot(t, acc_mag_smoothed)
    # ax[1].set_ylabel("Acceleration magnitude")

    # ax[2].plot(t, curvature)
    # ax[2].set_ylabel("Curvature")

    # plt.xlabel("Time (s)")
    # plt.suptitle(f"Day {day}, {source} -> {sink}, size {size}")

    # plt.savefig(
    #     f"../data/figures/02_08_curvature/{day}_{source}_{sink}_size_{size}.pdf"
    # )
    # plt.close()

    # curvature = (vx * ay - vy * ax) / (vx**2 + vy**2) ** 1.5

    return vel_mag_smoothed, curvature


def compute_velocity_and_acceleration(
    trajectory, window_length=None, polyorder=2
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the velocity and acceleration of a trajectory.

    Parameters
    ----------
    trajectory : np.ndarray
        Trajectory with shape (n_points, 2).
    window_length : int
        The length of the filter window
    polyorder : int
        The order of the polynomial to fit

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The velocity and acceleration of the trajectory
    """

    t = trajectory[:, 0]
    p = trajectory[:, 1:3]
    dt = np.diff(t)
    dp = np.diff(p, axis=0)
    v_forward = dp / dt[:, None]

    v_central = np.zeros_like(p)
    v_central[1:-1] = (p[2:] - p[:-2]) / (t[2:] - t[:-2])[:, None]
    v_central[0] = (p[1] - p[0]) / (t[1] - t[0])
    v_central[-1] = (p[-1] - p[-2]) / (t[-1] - t[-2])

    a = np.zeros_like(p)
    a[1:-1] = 2 * (v_forward[1:] - v_forward[:-1]) / (dt[1:] + dt[:-1])[:, None]
    a[0] = (v_central[1] - v_central[0]) / (t[1] - t[0])
    a[-1] = (v_central[-1] - v_central[-2]) / (t[-1] - t[-2])

    if window_length is not None:
        v_central = savgol_filter(
            v_central, window_length, polyorder, axis=0, mode="interp"
        )
        a = savgol_filter(a, window_length, polyorder, axis=0, mode="interp")

    return v_central, a


def compute_velocity_and_acceleration_numpy(
    trajectory,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the velocity and acceleration of a trajectory.

    Parameters
    ----------
    trajectory : np.ndarray
        Trajectory with shape (n_points, 2).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The velocity and acceleration of the trajectory
    """

    v = np.gradient(trajectory[:, 1:3], trajectory[:, 0], axis=0)
    a = np.gradient(v, trajectory[:, 0], axis=0)

    return v, a


def compute_curvature(trajectory, window_size) -> tuple[np.ndarray, np.ndarray]:
    """Compute the curvature of a trajectory.

    Parameters
    ----------
    trajectory : np.ndarray
        Trajectory with shape (n_points, 2).
    window_size : int
        The size of the window for smoothing the velocity and acceleration.

    Returns
    -------
    v_mag : np.ndarray
        Smoothed velocity magnitude.
    curvature : np.ndarray
        Curvature of the trajectory.
    """

    v, a = compute_velocity_and_acceleration(trajectory, window_size)
    # v, a = compute_velocity_and_acceleration_numpy(trajectory)

    # smooth the velocity and acceleration
    # v = savgol_filter(v, window_size, 2, axis=0)
    # a = savgol_filter(a, window_size, 2, axis=0)

    vx = v[:, 0]
    vy = v[:, 1]
    v_mag = np.linalg.norm(v, axis=1)

    ax = a[:, 0]
    ay = a[:, 1]

    curvature = (vx * ay - vy * ax) / (v_mag**3)

    return v_mag, curvature


def compute_osculating_circle(trajectory, window_size) -> tuple[np.ndarray, np.ndarray]:
    """Compute the center and radius of the osculating circle along a trajectory.

    Parameters
    ----------
    trajectory : np.ndarray
        Array of shape (n, 3), where the first column is time and the next two are (x, y).
    window_size : int
        The size of the window for smoothing the velocity and acceleration.

    Returns
    -------
    centers : np.ndarray
        The centers of the osculating circles (n, 2).
    radii : np.ndarray
        The radii of curvature (n,).
    """

    v, a = compute_velocity_and_acceleration(trajectory, window_size)

    vx = v[:, 0]
    vy = v[:, 1]
    v_mag = np.linalg.norm(v, axis=1)

    ax = a[:, 0]
    ay = a[:, 1]

    curvature = (vx * ay - vy * ax) / (v_mag**3)

    radius = np.where(
        np.abs(curvature) > 1e-8, 1 / curvature, np.inf
    )  # Handle zero curvature cases

    normal_direction = np.column_stack((-v[:, 1], v[:, 0]))
    normal_unit = normal_direction / np.maximum(
        np.linalg.norm(normal_direction, axis=1, keepdims=True), 1e-8
    )

    centers = trajectory[:, 1:3] + radius[:, None] * normal_unit

    return centers, radius


def hinge_function(x, y_hinge, x_hinge, slope):
    """Hinge function

    Parameters
    ----------
    x : np.ndarray
        The x values
    y_hinge : float
        The y value at the hinge point
    x_hinge : float
        The hinge point
    slope : float
        The slope of the hinge function

    Returns
    -------
    np.ndarray
        The hinge function
    """
    return np.where(x < x_hinge, y_hinge, y_hinge + slope * (x - x_hinge))


def fit_hinge_function(x, y, y_hinge=None, x_hinge=None, slope=None) -> tuple:
    """Fit a hinge function to the data

    Parameters
    ----------
    x : np.ndarray
        The x values
    y : np.ndarray
        The y values
    y_hinge : float
        The y value at the hinge point
    x_hinge : float
        The hinge point
    slope : float
        The slope of the hinge function

    Returns
    -------
    tuple
        The parameters of the hinge function
    """

    # find initial guess
    if y_hinge is None:
        y_hinge = np.nanmin(y)
    if x_hinge is None:
        x_hinge = np.nanmedian(x)
    if slope is None:
        slope = 0

    # remove NaN values
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    popt, pcov = curve_fit(hinge_function, x, y, p0=[y_hinge, x_hinge, slope])

    # compute R^2
    residuals = y - hinge_function(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return popt, r_squared


def plot_hinge_function(
    xmin, xmax, y_hinge, x_hinge, slope, ax, label=None, color="black"
):
    """Plot the hinge function

    Parameters
    ----------
    xmin : float
        The minimum x value
    xmax : float
        The maximum x value
    y_hinge : float
        The y value at the hinge point
    x_hinge : float
        The hinge point
    slope : float
        The slope of the hinge function
    ax : matplotlib.axes.Axes
        The axis to plot the hinge function
    label : str
        The label of the hinge function
    color : str
        The color of the hinge function
    """
    x = np.linspace(xmin, xmax, 100)
    y = hinge_function(x, y_hinge, x_hinge, slope)
    ax.plot(x, y, label=label, color=color, linewidth=2)
    ax.axvline(x_hinge, color=color, linestyle="dotted")
    ax.axhline(y_hinge, color=color, linestyle="dotted")


def piecewise_linear(x, x_hinge, y_hinge, slope1, slope2):
    """Piecewise linear function

    Parameters
    ----------
    x : np.ndarray
        The x values
    x_hinge : float
        The x-coordinate where the function transitions between slopes.
    y_hinge : float
        The y value at the hinge point.
    slope1 : float
        Slope for x < x_hinge.
    slope2 : float
        Slope for x >= x_hinge.

    Returns
    -------
    np.ndarray
        The piecewise linear function.
    """
    return np.where(
        x < x_hinge, y_hinge + slope1 * (x - x_hinge), y_hinge + slope2 * (x - x_hinge)
    )


def fit_piecewise_linear(x, y, x_hinge=None, y_hinge=None, slope1=None, slope2=None):
    """Fit a piecewise linear function to the data

    Parameters
    ----------
    x : np.ndarray
        The x values
    y : np.ndarray
        The y values
    x_hinge : float
        The x-coordinate where the function transitions between slopes.
    y_hinge : float
        The y value at the hinge point.
    slope1 : float
        Slope for x < x_hinge.
    slope2 : float
        Slope for x >= x_hinge.

    Returns
    -------
    tuple
        The parameters of the piecewise linear function.
    """
    # find initial guess
    if x_hinge is None:
        x_hinge = np.nanmedian(x)
    if y_hinge is None:
        y_hinge = np.nanmin(y)
    if slope1 is None:
        slope1 = 0
    if slope2 is None:
        slope2 = 0

    popt, pcov = curve_fit(
        piecewise_linear, x, y, p0=[x_hinge, y_hinge, slope1, slope2]
    )

    # compute R^2
    residuals = y - piecewise_linear(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return popt, r_squared


def plot_piecewise_linear(
    xmin, xmax, x_hinge, y_hinge, slope1, slope2, ax, label=None, color="black"
):
    """
    Plot a piecewise linear function with a hinge at x_hinge.

    Parameters
    ----------
    xmin : float
        The minimum x value.
    xmax : float
        The maximum x value.
    x_hinge : float
        The x value at the hinge point.
    y_hinge : float
        The y value at the hinge point
    slope1 : float
        Slope for x < x_hinge.
    slope2 : float
        Slope for x >= x_hinge.
    ax : matplotlib.axes.Axes
        The axis to plot the piecewise linear function.
    label : str
        The label of the piecewise linear function.
    color : str
        The color of the piecewise linear function.
    """
    x = np.linspace(xmin, xmax, 100)
    y = piecewise_linear(x, x_hinge, y_hinge, slope1, slope2)
    ax.plot(x, y, label=label, color=color, linewidth=2)
    ax.axvline(x_hinge, color=color, linestyle="dotted")
    ax.axhline(y_hinge, color=color, linestyle="dotted")


def compute_average_interpersonal_distance(
    trajectory_A: np.ndarray, trajectory_B: np.ndarray
) -> float:
    """Compute the distance between two position, at each time stamp.

    Parameters
    ----------
    trajectory_A : np.ndarray
        The trajectory of the first person
    trajectory_B : np.ndarray
        The trajectory of the second person

    Returns
    -------
    float
        The average interpersonal distance
    """

    trajectory_A_sim, trajectory_B_sim = compute_simultaneous_observations(
        [trajectory_A, trajectory_B]
    )

    distance = np.sqrt(
        (trajectory_A_sim[:, 1] - trajectory_B_sim[:, 1]) ** 2
        + (trajectory_A_sim[:, 2] - trajectory_B_sim[:, 2]) ** 2
    )

    return np.mean(distance)


def compute_angular_velocity(trajectory, window_size) -> np.ndarray:
    """Compute the angular velocity of a trajectory.

    Parameters
    ----------
    trajectory : np.ndarray
        Trajectory with shape (n_points, 2).
    window_size : int
        The size of the window for smoothing the velocity and acceleration.

    Returns
    -------
    np.ndarray
        Angular velocity of the trajectory.
    """
    velocity, _ = compute_velocity_and_acceleration(trajectory, window_size)
    vx = velocity[:, 0]
    vy = velocity[:, 1]

    theta = np.arctan2(vy, vx)
    theta = np.unwrap(theta)
    omega = np.diff(theta) / np.diff(trajectory[:, 0])
    omega = np.insert(omega, 0, omega[0])

    return omega


def compute_angular_velocity_old(trajectory, window_size) -> np.ndarray:
    """Compute the angular velocity of a trajectory.

    Parameters
    ----------
    trajectory : np.ndarray
        Trajectory with shape (n_points, 2).
    window_size : int
        The size of the window for smoothing the velocity and acceleration.

    Returns
    -------
    np.ndarray
        Angular velocity of the trajectory.
    """
    velocity = trajectory[:, 5:7] / 1000
    velocity_smoothed = savgol_filter(velocity, window_size, 2, axis=0)

    vx = velocity_smoothed[:, 0]
    vy = velocity_smoothed[:, 1]

    theta = np.arctan2(vy, vx)
    theta = np.unwrap(theta)
    omega = np.diff(theta) / np.diff(trajectory[:, 0])
    omega = np.insert(omega, 0, omega[0])

    return omega
