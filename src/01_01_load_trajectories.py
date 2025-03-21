import os
import matplotlib.pyplot as plt
import numpy as np

from parameters import DAYS, MIN_VEL, MAX_VEL, MIN_DURATION
from utils import pickle_load, pickle_save, compute_center_of_mass_trajectory


def compute_acceleration(trajectory):
    """Compute the acceleration from a trajectory

    Parameters
    ----------
    trajectory : np.ndarray
        The trajectory

    Returns
    -------
    np.ndarray
        The acceleration
    """

    ax = np.diff(trajectory[:, 5]) / np.diff(trajectory[:, 0])
    ax = np.append(ax, ax[-1])
    ay = np.diff(trajectory[:, 6]) / np.diff(trajectory[:, 0])
    ay = np.append(ay, ay[-1])

    return np.vstack((ax, ay)).T


def is_trajectory_valid(trajectory):
    """Check if a trajectory is valid (i.e. the average velocity is between MIN_VEL and MAX_VEL and the duration is at least MIN_DURATION)

    Parameters
    ----------
    trajectory : np.ndarray
        The trajectory to check

    Returns
    -------
    bool
        True if the trajectory is valid, False otherwise
    """

    vx = trajectory[:, 5]
    vy = trajectory[:, 6]
    mag_v = (vx**2 + vy**2) ** 0.5
    avg_v = mag_v.mean()

    duration = trajectory[-1, 0] - trajectory[0, 0]

    if avg_v < MIN_VEL or avg_v > MAX_VEL or duration < MIN_DURATION:
        return False

    return True


if __name__ == "__main__":

    data_dir = "../data/datasets"

    for environment in ["atc", "diamor"]:
        groups = []
        environment_dir = os.path.join(data_dir, environment)
        for day in DAYS[environment]:
            # load the trajectories
            trajectory_path = (
                os.path.join(environment_dir, f"trajectories_raw_{day}.pkl")
                if environment == "diamor"
                else os.path.join(environment_dir, f"trajectories_{day}.pkl")
            )
            daily_trajectories = pickle_load(trajectory_path)
            # load the individual annotations
            individual_annotations_path = os.path.join(
                environment_dir, f"individuals_annotations_{day}.pkl"
            )
            individual_annotations = pickle_load(individual_annotations_path)
            # load the group annotations
            groups_annotations_path = os.path.join(
                environment_dir, f"groups_annotations_{day}.pkl"
            )
            groups_annotations = pickle_load(groups_annotations_path)

            group_members = set()

            # iterate over the groups
            for group_id in groups_annotations:
                group_data = groups_annotations[group_id]

                soc_binding = (
                    group_data.get("soc_rel", None)
                    if environment == "atc"
                    else group_data.get("interaction", None)
                )

                group = {
                    "id": group_id,
                    "day": day,
                    "size": group_data["size"],
                    "members": [],
                    "soc_binding": soc_binding,
                }

                for member_id in group_data["members"]:
                    group_members.add(member_id)
                    if member_id not in daily_trajectories:
                        continue
                    member_trajectory = daily_trajectories[member_id]

                    if not is_trajectory_valid(member_trajectory):
                        continue

                    acc = compute_acceleration(member_trajectory)
                    member_trajectory = np.hstack((member_trajectory, acc))

                    member = {
                        "id": member_id,
                        "trajectory": member_trajectory,
                    }

                    group["members"].append(member)

                if len(group["members"]) != group["size"]:
                    # print(
                    #     f"Group {group_id} has {len(group['members'])} members but {group['size']} were expected"
                    # )
                    continue

                group["center_of_mass"] = compute_center_of_mass_trajectory(
                    [member["trajectory"] for member in group["members"]]
                )

                # fig, ax = plt.subplots(figsize=(12, 6))
                # for member in group["members"]:
                #     ax.plot(member["trajectory"][:, 1], member["trajectory"][:, 2])
                # ax.plot(
                #     group["center_of_mass"][:, 1], group["center_of_mass"][:, 2], "k"
                # )
                # plt.axis("equal")
                # plt.show()

                groups.append(group)

            # iterate over the individuals
            for pedestrian_id in daily_trajectories:
                if pedestrian_id in group_members:
                    continue

                pedestrian_trajectory = daily_trajectories[pedestrian_id]

                if pedestrian_id not in individual_annotations:
                    continue
                if not individual_annotations[pedestrian_id]["non_group"]:
                    continue

                if not is_trajectory_valid(pedestrian_trajectory):
                    continue

                acc = compute_acceleration(pedestrian_trajectory)
                pedestrian_trajectory = np.hstack((pedestrian_trajectory, acc))

                group = {
                    "id": pedestrian_id,
                    "day": day,
                    "size": 1,
                    "members": [
                        {
                            "id": pedestrian_id,
                            "trajectory": pedestrian_trajectory,
                        }
                    ],
                    "center_of_mass": pedestrian_trajectory,
                    "soc_binding": None,
                }

                groups.append(group)

        # show statistics
        print("Environment:", environment)
        group_sizes = sorted(set([group["size"] for group in groups]))
        for size in group_sizes:
            count = len([group for group in groups if group["size"] == size])
            print(f"Groups of size {size}: {count}")

        for day in DAYS[environment]:
            # save the groups
            groups_day = [group for group in groups if group["day"] == day]
            soc_bindings = set([group["soc_binding"] for group in groups_day])
            for soc_binding in soc_bindings:
                count = len(
                    [
                        group
                        for group in groups_day
                        if group["soc_binding"] == soc_binding
                    ]
                )
                print(f"Day {day}, soc_binding {soc_binding}: {count}")

        pickle_save(groups, f"../data/intermediate/01_01_groups_{environment}.pkl")
