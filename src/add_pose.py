
import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):
    # TODO: Add the odometry factor between X(4) and X(5) to the graph (BetweenFactorPose2)

    # TODO: Based on the odometry, find the initial estimate for the pose of X(5) and add it to the graph
    
    pose_3 = initial_estimate.atPose2(X(3))

    # Target global position for X(4)
    pose_4 = gtsam.Pose2(4.0 + math.sqrt(2), math.sqrt(2), math.pi / 2)

    # Compute odometry as relative transform from X(3) to X(4)
    odometry = pose_3.between(pose_4)

    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), odometry, ODOMETRY_NOISE))
    initial_estimate.insert(X(4), pose_4)

    return graph, initial_estimate

pose_3 = gtsam.Pose2(4.0, 0.0, 0.0)
odometry = gtsam.Pose2(np.sqrt(2), np.sqrt(2), np.pi / 2)
pose_4 = pose_3.compose(odometry)
print(f"X(4): x={pose_4.x():.4f}, y={pose_4.y():.4f}, theta={pose_4.theta():.4f}")
print(f"Expected: x={4.0 + math.sqrt(2):.4f}, y={math.sqrt(2):.4f}, theta={math.pi/2:.4f}")