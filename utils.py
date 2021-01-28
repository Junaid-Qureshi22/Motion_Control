from __future__ import division
import numpy as np
from scipy.special import fresnel
import sys
import yaml
import math
import scipy.linalg


def angle_diff(a, b):
    # Modulo 2*pi of the difference between two angles
    angles = np.unwrap([a, b])
    return angles[0] - angles[1]


def distance(pose_1, pose_2):
    return math.sqrt((pose_1[0] - pose_2[0])**2 + (pose_1[1] - pose_2[1])**2)


def projection_on_path(path, pose, path_index, path_size):
    index_window = 100
    R2 = np.clip(path_index + index_window, 0, path_size)
    R1 = np.clip(path_index, 0, path_size)
    d = []
    for i in range(R1, R2):
        d.append(distance(path[i], pose))
    min_index = np.argmin(d) + R1
    return path[min_index], min_index


def find_goal_point(path, pose, min_index, L, path_size):
    index_window = 200
    R1 = min_index
    R2 = np.clip(min_index + index_window, 0, path_size)
    err = []
    for i in range(R1, R2):
        error = abs(distance(path[i], pose) - L)
        err.append(error)
    goal_index = np.argmin(err) + R1
    return path[goal_index], goal_index


def side(heading, pose, proj):
    tangent_heading = [np.cos(heading), np.sin(heading)]
    tangent_normal = [pose[0] - proj[0], pose[1] - proj[1]]
    crossp = np.cross(tangent_heading, tangent_normal)
    if crossp > 0:
        side = -1
    else:
        side = 1
    return side


def get_lateral_error(path, index, theta, pose, path_size):
    previous_index = np.clip(0, index - 1, path_size)
    if index >= path_size - 1:
        next_index = index
    else:
        next_index = index + 1
    prev_pose = path[previous_index, :]
    next_pose = path[next_index, :]
    norm = distance(next_pose, prev_pose)
    # Split [prev_pose, next_pose] line to get more precise value
    t = np.linspace(0, 1, 100)
    points_x = prev_pose[0] + t*norm*np.cos(theta)
    points_y = prev_pose[1] + t*norm*np.sin(theta)
    points = np.array(list(zip(points_x, points_y)))
    d = []
    for i in range(len(points)):
        d.append(distance(points[i], pose))
    lateral_error = min(d)
    if lateral_error < 0.0001:
        lateral_error = 0
    return lateral_error


def heading_error(path_heading, vehicle_heading):
    error = -(vehicle_heading - path_heading)
    error = np.arctan2(np.sin(error), np.cos(error))
    return error


def steering_limitation(u, limit):
    return np.clip(u, -limit, limit)


def get_steering_rate_limitation(max_steering_rate_ul, max_steering_rate_ll, steering_speed_ul, steering_speed_ll, speed):
    if speed <= steering_speed_ll:
        limit = max_steering_rate_ul
    elif speed >= steering_speed_ul:
        limit = max_steering_rate_ll
    else:
        limit = max_steering_rate_ul + ((max_steering_rate_ul - max_steering_rate_ll) / (steering_speed_ll - steering_speed_ul)) * (speed - steering_speed_ll)
    return limit


def steering_rate_limitation(u, u_previous, limit,Ts):
    return np.clip(u, u_previous - limit * Ts, u_previous + limit * Ts)


def get_delta(theta, pose, path, L, b):
    goal_vector = path - pose
    norm_goal_vector = distance(path, pose)
    heading_vector = np.array([np.cos(theta), np.sin(theta)])
    crossp = np.cross(goal_vector, heading_vector)
    ## cap the domain of acos(x) between -1 and 1
    limit_domain = np.clip(np.dot(goal_vector, heading_vector) / norm_goal_vector, -1, 1)
    ## Including the sign depending on which side the vehicle is
    alpha = -np.sign(crossp) * np.arccos(limit_domain)
    delta = math.atan2(2*b*np.sin(alpha), L)
    return delta


def compute_lqr_gains(lqr_model):
# - linear quadratic regulator control gives us the state feeback control gains which minimize the quadratic cost function.
# - The minimum point is located by solving the algebric Riccati equation
    A = lqr_model.get_state_matrix()
    B = lqr_model.get_control_matrix()
    Q, R = lqr_config_parameters()
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R)) # - solution of Riccati equation
    K = np.dot(np.linalg.inv(R),(np.dot(np.transpose(B),X))) # - state feedback gain matrix
    return K[0, 0], K[0, 1]

def lqr_config_parameters():
    Q = [[1, 0], [0, 1]] # - penalty matrix w.r.t state
    R = [[10]] # - penalty matrix w.r.t control
    return Q, R

def zoh(Ts, Te, index):
    return (index % (Te / Ts) == 0)


def sign_curvature(path_heading):
    if path_heading >= -np.pi / 2 and path_heading <= np.pi / 2:
        sign = 1
    elif path_heading > np.pi / 2 or path_heading < -np.pi / 2:
        sign = -1
    return sign
