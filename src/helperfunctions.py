import gtsam
import math
from gtsam import Pose2, Point2, Rot2, BearingRangeFactor2D


def add_pose_from_global(
    graph,
    initial_estimate,
    prev_key,
    new_key,
    prev_pose,
    new_pose_global,
    odom_noise
):
    """
    Adds a new pose using global coordinates by automatically
    computing the correct BetweenFactor.
    """

    # Compute relative motion in prev_pose frame
    # relative_pose = new_pose_global.between(prev_pose)
    relative_pose = prev_pose.between(new_pose_global)

    # Add odometry factor
    graph.add(
        gtsam.BetweenFactorPose2(
            prev_key,
            new_key,
            relative_pose,
            odom_noise
        )
    )

    # Insert initial estimate
    initial_estimate.insert(new_key, new_pose_global)

    return graph, initial_estimate


def add_landmark_measurement_from_global(
    graph,
    pose_key,
    pose,
    landmark_key,
    landmark_point,
    measurement_noise,
    add_factor=True
):
    """
    Adds a bearing-range measurement computed from global positions.
    """

    # Compute bearing and range automatically
    bearing = pose.bearing(landmark_point)
    distance = pose.range(landmark_point)
    # print("Angle in degrees:", math.degrees(bearing.theta()))
    # print("Adding measurement: bearing =", bearing.theta(), " rad, range = ", distance," m")
    if add_factor:
        graph.add(
            BearingRangeFactor2D(
                pose_key,
                landmark_key,
                bearing,
                distance,
                measurement_noise
            )
        )
        return graph
    return distance, bearing.theta()
