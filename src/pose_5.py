import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)


def add_pose(graph, initial_estimate, pose_5):
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate


def add_landmark_measurement(graph, result, pose_5, landmark):
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph


def optimize(graph, initial_estimate):
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    print(f"Error: {graph.error(result):.4f}")
    return result


def minimize_marginals(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    best_score = float("inf")
    sum_of_marginals = float("inf")

    for pose_label, pose_5 in pose_options.items():
        for lm_idx in [1, 2]:
            g   = gtsam.NonlinearFactorGraph(graph)
            est = gtsam.Values(initial_estimate)

            g, est = add_pose(g, est, pose_5)
            result = optimize(g, est)
            g = add_landmark_measurement(g, result, pose_5, lm_idx)
            result = optimize(g, est)

            marginals = gtsam.Marginals(g, result)

            # Use observed landmark covariance for selection
            selection_score = marginals.marginalCovariance(L(lm_idx)).sum()

            # Sum over all landmarks for the return value
            total = 0.0
            for j in [1, 2]:
                total += marginals.marginalCovariance(L(j)).sum()

            print(f"Pose {pose_label}, L({lm_idx}): sum of marginals = {selection_score:.4f}")

            if selection_score < best_score:
                best_score = selection_score
                sum_of_marginals = total
                best_pose = pose_label
                best_landmark = lm_idx

    print(f"\nBest: pose='{best_pose}', landmark=L({best_landmark}), sum={sum_of_marginals:.4f}")
    return best_pose, best_landmark, sum_of_marginals


def minimize_errors(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    best_score = float("inf")
    sum_of_errors = float("inf")

    for pose_label, pose_5 in pose_options.items():
        for lm_idx in [1, 2]:
            g   = gtsam.NonlinearFactorGraph(graph)
            est = gtsam.Values(initial_estimate)

            g, est = add_pose(g, est, pose_5)
            result = optimize(g, est)
            g = add_landmark_measurement(g, result, pose_5, lm_idx)
            result = optimize(g, result)  # use result as initial estimate

            # Use pose marginal covariance traces for selection
            marginals = gtsam.Marginals(g, result)
            list_of_errors = []
            for i in [1, 2, 3]:
                cov = marginals.marginalCovariance(X(i))
                list_of_errors.append(np.trace(cov))

            selection_score = sum(list_of_errors)
            total_error = g.error(result)

            print(f"Pose {pose_label}, L({lm_idx}): sum of errors = {selection_score:.6f}")

            if selection_score < best_score:
                best_score = selection_score
                sum_of_errors = total_error
                best_pose = pose_label
                best_landmark = lm_idx

    print(f"\nBest: pose='{best_pose}', landmark=L({best_landmark}), sum={sum_of_errors:.2e}")
    return best_pose, best_landmark, sum_of_errors