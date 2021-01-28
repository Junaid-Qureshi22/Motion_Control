import numpy as np
import utils
import models
import pytest
import math
from pytest import approx

line_size = 10000
curve_size = 10000
L = 2   # look ahead distance
b = 1.4 # wheel_base
V = 3.5 # speed
lf = 0.7
lr = 0.7
C = 340
m = 550
Iz = 3200
dynamic_parameters = np.array([lf, lr, C, m, Iz])
K = 1 # steady state gain for vehicle steering behaviour
tau = 0.1 # first order time constant for vehicle steering behaviour
td = 0.2 # time delay of first order model
tc = 0.2 # control steering delay
dt = 0.01 # sampling time
t_end = 50.0 # simulation Time
max_steering_rate_ul = 0.3 # upper limit for maximum steering rate
max_steering_rate_ll = 0.1 # lower limit for maximum steering rate
max_steering_rate_vehicle_min_speed = 2 # lower threshold of vehicle speed for maximum steering rate
max_steering_rate_vehicle_max_speed = 5 # upper threshold of vehicle speed for maximum steering rate
beta_i = 0.0
yaw_rate_i = 0.0
x_i = 1
y_i = 2
theta_i = 0.5

def line():
    x = np.linspace(0, 100, line_size)
    y = x
    y1 = np.ones(line_size)
    curvature = np.zeros(line_size)
    C = np.array(list(zip(x, y)))
    return C, x, y, curvature, np.arctan(y1)


def curve():
    x = np.linspace(0, 100, curve_size)
    y = np.zeros(curve_size)
    y1 = np.zeros(curve_size)
    curvature = np.zeros(curve_size)
    theta = np.zeros(curve_size)
    y2 = 0.02
    for i in range(curve_size):
        y[i] = 0.1 * x[i]**2
        y1[i] = 0.2 * x[i]
        curvature[i] = abs(y2)/pow((1 + y1[i]**2), 1.5)
        theta[i] = math.atan2(y1[i], 1)
    C = np.array(list(zip(x, y)))
    return C, x, y, curvature, theta


pose_line, X_line, Y_line, curvature_line, heading_line = line()
pose_curve, X_curve, Y_curve, curvature_curve, heading_curve = curve()
point = [0.1, 0.2]


def test_projection_on_path():
    proj_line_expected = [0.15, 0.15]
    proj_curve_expected = [0.10414, 0.001084]
    proj_line, proj_line_index = utils.projection_on_path(pose_line, point, 0, line_size)
    proj_curve, proj_curve_index = utils.projection_on_path(pose_curve, point, 0, curve_size)
    assert proj_line[0] == approx(proj_line_expected[0], abs = 1e-3)
    assert proj_line[1] == approx(proj_line_expected[1], abs = 1e-3)
    assert proj_curve[0] == approx(proj_curve_expected[0], abs = 1e-2)
    assert proj_curve[1] == approx(proj_curve_expected[1], abs = 1e-3)


def test_find_goal_point():
    goal_line_expected = [1.5633, 1.5633]
    goal_curve_expected = [2.08612, 0.435189]
    _, proj_line_index = utils.projection_on_path(pose_line, point, 0, line_size)
    _, proj_curve_index = utils.projection_on_path(pose_curve, point, 0, curve_size)
    goal_line, goal_line_index = utils.find_goal_point(pose_line, point, proj_line_index, L, line_size)
    goal_curve, goal_curve_index = utils.find_goal_point(pose_curve, point, proj_curve_index, L, curve_size)
    assert goal_line[0] == approx(goal_line_expected[0], abs = 1e-2)
    assert goal_line[1] == approx(goal_line_expected[1], abs = 1e-2)
    assert goal_curve[0] == approx(goal_curve_expected[0], abs = 1e-2)
    assert goal_curve[1] == approx(goal_curve_expected[1], abs = 1e-2)


def test_side():
    side_line_expected = -1
    side_curve_expected = -1
    theta = 1.5708
    proj_line, _ = utils.projection_on_path(pose_line, point, 0, line_size)
    proj_curve, _ = utils.projection_on_path(pose_curve, point, 0, curve_size)
    side_line = utils.side(theta, point, proj_line)
    side_curve = utils.side(theta, point, proj_curve)
    assert side_line ==  side_line_expected
    assert side_curve ==  side_curve_expected


def test_get_lateral_error():
    lateral_error_line_expected = 0.0707
    lateral_error_curve_expected = 0.19896
    _, proj_line_index = utils.projection_on_path(pose_line, point, 0, line_size)
    _, proj_curve_index = utils.projection_on_path(pose_curve, point, 0, curve_size)
    theta_line = heading_line[proj_line_index]
    theta_curve = heading_curve[proj_curve_index]
    lateral_error_line = utils.get_lateral_error(pose_line, proj_line_index, theta_line, point, line_size)
    lateral_error_curve = utils.get_lateral_error(pose_curve, proj_curve_index, theta_curve, point, curve_size)
    assert lateral_error_line == approx(lateral_error_line_expected, abs = 1e-2)
    assert lateral_error_curve == approx(lateral_error_curve_expected, abs = 1e-2)


def test_get_delta():
    delta_line_expected = -0.7974
    delta_curve_expected = -0.94722
    theta = 1.5708
    _, proj_line_index = utils.projection_on_path(pose_line, point, 0, line_size)
    _, proj_curve_index = utils.projection_on_path(pose_curve, point, 0, curve_size)
    goal_line, _ = utils.find_goal_point(pose_line, point, proj_line_index, L, line_size)
    goal_curve, _ = utils.find_goal_point(pose_curve, point, proj_curve_index, L, curve_size)
    delta_line = utils.get_delta(theta, point, goal_line, L, b)
    delta_curve = utils.get_delta(theta, point, goal_curve, L, b)
    assert delta_line == approx(delta_line_expected, abs = 1e-2)
    assert delta_curve == approx(delta_curve_expected, abs = 1e-2)


def test_compute_lqr_gains():
    lqr_model = models.Model(V, b)
    K1_expected = 0.31622
    K2_expected = 0.9927
    K1, K2 = utils.compute_lqr_gains(lqr_model)
    assert K1 == approx(K1_expected, abs = 1e-3)
    assert K2 == approx(K2_expected, abs = 1e-3)


def test_first_order_model():
    phi_initial = 0
    steering_model = models.DiscreteFirstOrderModel(K, tau, dt, td)
    u = np.zeros(100)
    phi = np.zeros(100)
    u[10:] = 10
    for i in range(100):
        phi[i] = steering_model.get_output(u[i])
    assert phi[39] == approx(6.32, abs = 1e-2)


def test_get_steering_rate_limitation():
    s1 = 1
    s2 = 3
    s3 = 5.5
    ms1_expected = 0.3
    ms2_expected = 0.233333
    ms3_expected = 0.1
    ms1 = utils.get_steering_rate_limitation(max_steering_rate_ul, max_steering_rate_ll, max_steering_rate_vehicle_max_speed, max_steering_rate_vehicle_min_speed, s1)
    ms2 = utils.get_steering_rate_limitation(max_steering_rate_ul, max_steering_rate_ll, max_steering_rate_vehicle_max_speed, max_steering_rate_vehicle_min_speed, s2)
    ms3 = utils.get_steering_rate_limitation(max_steering_rate_ul, max_steering_rate_ll, max_steering_rate_vehicle_max_speed, max_steering_rate_vehicle_min_speed, s3)
    assert ms1 == approx(ms1_expected, abs = 1e-3)
    assert ms2 == approx(ms2_expected, abs = 1e-3)
    assert ms3 == approx(ms3_expected, abs = 1e-3)


def test_steering_rate_limitation():
    u1 = 0.2
    u2 = 0.191
    u1_expected = 0.1925
    u2_expected = 0.191
    u_previous = 0.19
    limit = 0.25
    u1 = utils.steering_rate_limitation(u1, u_previous, limit, dt)
    u2 = utils.steering_rate_limitation(u2, u_previous, limit, dt)
    assert u1 == approx(u1_expected, abs = 1e-3)
    assert u2 == approx(u2_expected, abs = 1e-3)


def test_speed_controller():
    lateral_error_limit_speed = 0.5
    lateral_error_upper_limit = 0.5
    lateral_error_lower_limit = 0.2
    global_speed_target = 2.25
    acceleration = 0.5
    current_speed = 4
    lateral_error1 = 0.1
    lateral_error2 = 0.3
    lateral_error3 = 0.6
    speed_target_expected1 = 2.25
    speed_target_expected2 = 1.6667
    speed_target_expected3 = 0.5
    speed_controller = models.SpeedController(lateral_error_limit_speed, lateral_error_upper_limit, lateral_error_lower_limit, acceleration, global_speed_target, dt)
    speed_target1 = speed_controller.get_vehicle_speed_wrt_lateral_error(lateral_error1)
    speed_target2 = speed_controller.get_vehicle_speed_wrt_lateral_error(lateral_error2)
    speed_target3 = speed_controller.get_vehicle_speed_wrt_lateral_error(lateral_error3)
    assert speed_target1 == approx(speed_target_expected1, abs = 1e-4)
    assert speed_target2 == approx(speed_target_expected2, abs = 1e-4)
    assert speed_target3 == approx(speed_target_expected3, abs = 1e-4)
    speed1_expected = 3.995
    speed2_expected = 3.995
    speed3_expected = 3.995
    speed1 = speed_controller.get_vehicle_speed_wrt_acceleration(speed_target1, current_speed)
    speed2 = speed_controller.get_vehicle_speed_wrt_acceleration(speed_target2, current_speed)
    speed3 = speed_controller.get_vehicle_speed_wrt_acceleration(speed_target3, current_speed)
    assert speed1 == approx(speed1_expected, abs = 1e-4)
    assert speed2 == approx(speed2_expected, abs = 1e-4)
    assert speed3 == approx(speed3_expected, abs = 1e-4)


def test_state_estimation():
    speed = 5
    pose = [1, 2]
    theta = np.pi / 2
    phi = 0.1
    predicted_pose_expected = [0.96418, 2.9991]
    state_estimator = models.StateEstimation(tc, b)
    predicted_pose = state_estimator.get_state_estimation(speed, phi, pose, theta)
    assert predicted_pose[0] == approx(predicted_pose_expected[0], abs = 1e-3)
    assert predicted_pose[1] == approx(predicted_pose_expected[1], abs = 1e-3)
    phi = 1e-7
    predicted_pose_expected = [1, 3]
    predicted_pose = state_estimator.get_state_estimation(speed, phi, pose, theta)
    assert predicted_pose[0] == predicted_pose_expected[0]
    assert predicted_pose[1] == predicted_pose_expected[1]
    theta = np.pi / 4
    phi = 0.1
    predicted_pose_expected = [1.6812, 2.7318]
    predicted_pose = state_estimator.get_state_estimation(speed, phi, pose, theta)
    assert predicted_pose[0] == approx(predicted_pose_expected[0], abs = 1e-3)
    assert predicted_pose[1] == approx(predicted_pose_expected[1], abs = 1e-3)
    phi = 1e-7
    predicted_pose_expected = [1.7071, 2.7071]
    predicted_pose = state_estimator.get_state_estimation(speed, phi, pose, theta)
    assert predicted_pose[0] == approx(predicted_pose_expected[0], abs = 1e-3)
    assert predicted_pose[1] == approx(predicted_pose_expected[1], abs = 1e-3)


def test_kinematic_model():
    phi = 0.15
    beta_expected = 0.07542
    theta_dot_expected = 0.3778
    x_expected = 1.0307
    y_expected = 2.0168
    theta_expected = 0.503778
    state_evolution = models.KinematicModel(x_i, y_i, theta_i, dt)
    beta, theta_dot = state_evolution.get_beta_and_psi_dot(phi, b, V)
    x = state_evolution.get_x_updated(V, theta_i, beta_i)
    y = state_evolution.get_y_updated(V, theta_i, beta_i)
    theta = state_evolution.get_theta_updated(theta_dot)
    assert beta == approx(beta_expected, abs = 1e-3)
    assert theta_dot == approx(theta_dot_expected, abs = 1e-3)
    assert x == approx(x_expected, abs = 1e-3)
    assert y == approx(y_expected, abs = 1e-3)
    assert theta == approx(theta_expected, abs = 1e-3)


def test_dynamic_model():
    phi = 0.15
    beta_dot_expected = 0.0265
    beta_expected = 0.000265
    theta_dot_expected = 0.0001115
    x_expected = 1.0307
    y_expected = 2.0168
    theta_expected = 0.5000011
    state_evolution = models.DynamicModel(dynamic_parameters, x_i, y_i, theta_i, beta_i, yaw_rate_i, dt)
    beta, theta_dot = state_evolution.get_beta_and_psi_dot(phi, b, V)
    x = state_evolution.get_x_updated(V, theta_i, beta_i)
    y = state_evolution.get_y_updated(V, theta_i, beta_i)
    theta = state_evolution.get_theta_updated(theta_dot)
    assert beta == approx(beta_expected, abs = 1e-3)
    assert theta_dot == approx(theta_dot_expected, abs = 1e-3)
    assert x == approx(x_expected, abs = 1e-3)
    assert y == approx(y_expected, abs = 1e-3)
    assert theta == approx(theta_expected, abs = 1e-3)
