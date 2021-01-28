from __future__ import division
import numpy as np
from scipy import signal
import math
import matplotlib.pyplot as plt
import collections
import sys
import utils
import trajectory
import models
import yaml
import csv
from numpy import asarray
from numpy import savetxt
## Brief
# This script is a path following simulator for different vehicles.
# -- The files "<vehicle>.yaml" contain the parameters which are dependent to
# the plaform. The type of controller and its parameters must be specified in
# this files.
# -- The file configuration.yaml contains all other configurable
# parameters (simulation, vehicle starting pose, type of path, etc.)
# -- The script can be run with --> $ python path_following.py <vehicle>.yaml

#####################################
## Get Configuration
#####################################

init_file = sys.argv[1]

## Get vehicle dependent parameter
with open(init_file) as f:
    params = yaml.safe_load(f)
vehicle_model = params["model"]["vehicle_model"]
wheelbase = params["dimension"]["wheelbase"]
steering_delay = params["delay"]["steering_delay"]
steering_max = params["steering"]["steering_max"]
max_steering_rate_upper_limit = params["steering"]["max_steering_rate_upper_limit"]
max_steering_rate_lower_limit = params["steering"]["max_steering_rate_lower_limit"]
max_steering_rate_vehicle_min_speed = params["steering"]["max_steering_rate_vehicle_min_speed"]
max_steering_rate_vehicle_max_speed = params["steering"]["max_steering_rate_vehicle_max_speed"]

control_steering_delay = params["steering"]["control_steering_delay"]
L = params["Look_ahead_distance"]["L"]
type = params["controller"]["type"]
Kp = params["controller"]["Kp"]        # proportional gain
Ki = params["controller"]["Ki"]        # integral gain
Kd = params["controller"]["Kd"]        # derivative gain
Ks = params["controller"]["Ks"]        # stanlay controller gain
steering_gain_constant = params["vehicle_steering_behaviour"]["K"]        # steady state gain
steering_time_constant = params["vehicle_steering_behaviour"]["tau"]        # first order time constant
dynamic_parameters = np.array([params["dynamic_model_parameters"]["lf"], params["dynamic_model_parameters"]["lr"],
                               params["dynamic_model_parameters"]["C"], params["dynamic_model_parameters"]["m"],
                               params["dynamic_model_parameters"]["Iz"]]) # array containing dybamic model parameters

## Vehicle configuration
with open("configuration.yaml") as f:
    params = yaml.safe_load(f)

## Vehicle Initial conditions
x_r_init = params["pose"]["x"]
y_r_init = params["pose"]["y"]
V_init = params["speed"]["initial_speed"]
global_speed_target = params["speed"]["global_speed_target"]
accel = params["acceleration"]["acceleration"]
reverse = False

## Simulation configuration
dt = params["simulation"]["dt"]
t_end = params["simulation"]["t_end"]  # end of simulation
Ts = params["simulation"]["Ts"]        # Control input sampling time

## Lateral error limitation lqr_config_parameters
lateral_error_limit_speed = params["lateral_error_limitation"]["lateral_error_limit_speed"]
lateral_error_upper_limit = params["lateral_error_limitation"]["lateral_error_upper_limit"]
lateral_error_lower_limit = params["lateral_error_limitation"]["lateral_error_lower_limit"]
#####################################
## Define path that will be used
#####################################
ans=True
while ans:
    print("""
    1. line
    2. Hermite_curve
    3. Hermite_right_turn
    4. Hermite_left_turn
    5. quadratic_bezier_curve
    6. cubic_bezier_curve
    7. Loop1
    8. Loop2
    """)
    ans = raw_input("select the path ")
    if ans == "1":
      print("path = line")
      path = trajectory.LinePath()
      theta_init = 0.7854
      break
    elif ans == "2":
      print("path = Hermite_curve_one")
      path = trajectory.HermitePath(1, 1)
      theta_init = 1.372
      break
    elif ans == "3":
      print("path = Hermite_right_turn")
      path = trajectory.HermitePath(2, 1)
      theta_init = 1.353
      break
    elif ans == "4":
      print("path = Hermite_left_turn")
      path = trajectory.HermitePath(2, -1)
      theta_init = 1.789
      break
    elif ans == "5":
      print("path = quadratic_bezier_curve")
      path = trajectory.BezierCurve(2)
      theta_init = 1.245
      break
    elif ans == "6":
      print("path = cubic_bezier_curve")
      path = trajectory.BezierCurve(3)
      theta_init = 1.265
      break
    elif ans=="7":
      print("path = Loop1")
      path = trajectory.Loop1()
      theta_init = 0.50545
      break
    elif ans=="8":
       print("path = Loop2")
       path = trajectory.Loop2()
       theta_init = 0.31567
       break
ref_pose, ref_X, ref_Y, ref_curvature, heading, path_size = path.get_path_information()

#####################################
## Set simulation parameters
#####################################
print("model = {}".format(vehicle_model))
print ("controller = {}".format(type))
if (type == "p"):
    print ("Kp = {}".format(Kp))
elif (type == "pd" or type == "pd_f" or type == "pd_v" or type == "pd_v_f"):
    print ("Kp = {}".format(Kp))
    print ("Kd = {}".format(Kd))
elif (type == "pid" or type == "pid_f"):
    print ("Kp = {}".format(Kp))
    print ("Ki = {}".format(Ki))
    print ("Kd = {}".format(Kd))
elif (type == "pp" or type == "pp_f"):
    print("Look_ahead_distance = {}".format(L))
elif (type == "stanlay"):
    print("Ks = {}".format(Ks))
elif (type == "lqr" or type == "lqr_f"):
    lqr_model = models.Model(global_speed_target, wheelbase)
    K1, K2 = utils.compute_lqr_gains(lqr_model)
    print("K1 = {}".format(K1))
    print("K2 = {}".format(K2))
else:
    raise Exception('Wrong controller type: {}', type)

# define the lists
x = []
y = []
theta = []
error = []
theta_error = []
u = []
phi = []
steering_rate = []
speed_target = []
vehicle_speed = []
path_heading = []
path_curvature = []
ff = []

# Initialize the system variables
x_current = x_r_init
y_current = y_r_init
theta_current = theta_init
current_vehicle_speed = V_init
phi_current = 0
beta_current = 0
yaw_rate_current = 0
actual_steering_rate = 0

# Initialize the other variables
heading_ise = 0.0
lateral_ise = 0.0
integral = 0
previous_proj_index = 0
limiting_index = 0
u_previous = 0
u_last = 0
i = 0

# Instantiate the models and controllers
steering_model = models.DiscreteFirstOrderModel(steering_gain_constant, steering_time_constant, dt, steering_delay)
speed_controller = models.SpeedController(lateral_error_limit_speed, lateral_error_upper_limit, lateral_error_lower_limit, accel, global_speed_target, dt)
state_estimator = models.StateEstimation(control_steering_delay, wheelbase)

#####################################
## Simulation loop
#####################################

# Vehicle course
integral = 0
previous_proj_index = 0
if vehicle_model == "kinematic":
    state_evolution = models.KinematicModel(x_r_init, y_r_init, theta_init, dt)
elif vehicle_model == "dynamic":
    state_evolution = models.DynamicModel(dynamic_parameters, x_r_init, y_r_init, theta_init, beta_current, yaw_rate_current, dt)
else:
    raise Exception('Wrong model: {}', vehicle_model)
while limiting_index < path_size:

    # get the vehicle position
    current_pose = [x_current, y_current]

    # get predicted position
    predicted_pose = state_estimator.get_state_estimation(current_vehicle_speed, phi_current, current_pose, theta_current)

    # append the state variables
    x.append(x_current)
    y.append(y_current)
    theta.append(theta_current)
    vehicle_speed.append(current_vehicle_speed)

    # compute errors
    proj_point, proj_index = utils.projection_on_path(ref_pose, predicted_pose, previous_proj_index, path_size)
    proj_heading = heading[proj_index]
    lateral_error = utils.get_lateral_error(ref_pose, proj_index, proj_heading, predicted_pose, path_size)
    signed_lateral_error = lateral_error * utils.side(proj_heading, predicted_pose, proj_point)
    heading_error = utils.heading_error(proj_heading, theta_current)
    proj_curvature = ref_curvature[proj_index]
    limiting_index = proj_index + 1
    if (type == "pp" or type == "pp_f"):
        goal_point, goal_index = utils.find_goal_point(ref_pose, current_pose, proj_index, L, path_size)
        goal_heading = heading[goal_index]
        delta = utils.get_delta(theta_current, current_pose, goal_point, L, wheelbase)
        proj_curvature = ref_curvature[goal_index]
        limiting_index = goal_index + 1
    path_heading.append(proj_heading)
    path_curvature.append(abs(proj_curvature))
    previous_proj_index = proj_index
    # get speed target w.r.t lateral error limitation
    speed_target = speed_controller.get_vehicle_speed_wrt_lateral_error(lateral_error)

    if reverse:
        heading_error = utils.angle_diff(heading_error, np.pi)

    # check if reverse
    if abs(heading_error) > 1.65:
        reverse = not reverse
        current_vehicle_speed = -current_vehicle_speed

    # compute ise errors
    heading_ise += heading_error**2
    lateral_ise += lateral_error**2
    # append the errors
    error.append(signed_lateral_error)
    theta_error.append(heading_error)

    ## Control law with zoh and delay
    if utils.zoh(dt, Ts, i):
        # Compute command at sampling time Ts
        integral = integral + signed_lateral_error*Ts
        feedforward = utils.sign_curvature(proj_heading) * np.arctan(proj_curvature * wheelbase)
        if (type == "p"):
            u_k = Kp*signed_lateral_error
        elif (type == "pd"):
            u_k = Kp*signed_lateral_error + Kd*heading_error
        elif (type == "pd_f"):
            u_k = feedforward + Kp*signed_lateral_error + Kd*heading_error
        elif (type == "pd_v"):
            u_k = Kp*signed_lateral_error + current_vehicle_speed * Kd * heading_error
        elif (type == "pd_v_f"):
            u_k = feedforward + Kp*signed_lateral_error + abs(current_vehicle_speed) * Kd * heading_error
        elif (type == "pid"):
            u_k = Kp*signed_lateral_error + Ki * integral + Kd * heading_error
        elif (type == "pid_f"):
            u_k = feedforward + Kp*signed_lateral_error + Ki * integral + Kd * heading_error
        elif (type == "pp"):
            u_k = delta
        elif (type == "pp_f"):
            u_k = feedforward + delta
        elif (type == "stanlay"):
            u_k = heading_error + math.atan2(Ks * signed_lateral_error, current_vehicle_speed)
        elif (type == "lqr"):
            u_k = K1 * signed_lateral_error + K2 * heading_error
        elif (type == "lqr_f"):
            u_k = feedforward + K1 * signed_lateral_error + K2 * heading_error
        else:
            raise Exception('Wrong controller type: {}', type)

        if reverse:
            u_k = -u_k

        u_current = u_k
        u_previous = u_current

    else:
        # Hold the previous command
        u_current = u_previous
    # Appply steering limitation
    u_current = utils.steering_limitation(u_current, steering_max)
    # get steering rate limitation
    steering_rate_limit = utils.get_steering_rate_limitation(max_steering_rate_upper_limit, max_steering_rate_lower_limit, max_steering_rate_vehicle_max_speed, max_steering_rate_vehicle_min_speed, current_vehicle_speed)
    # Apply steering rate limitation
    u_current = utils.steering_rate_limitation(u_current, u_last, steering_rate_limit, Ts)
    u_last = u_current

    # append the control variables
    u.append(u_current)
    phi.append(phi_current)
    steering_rate.append(actual_steering_rate)
    #print(i, proj_index, beta_current, yaw_rate_current)

    # get system updates
    phi_updated = steering_model.get_output(u_current) # - actual steering angle of vehicle considering first order response
    actual_steering_rate = (phi_updated - phi_current) / dt
    # get speed update w.r.t longitunal acceleration
    vehicle_speed_updated = speed_controller.get_vehicle_speed_wrt_acceleration(speed_target, current_vehicle_speed)
    beta_updated, yaw_rate_updated = state_evolution.get_beta_and_psi_dot(phi_current, wheelbase, current_vehicle_speed)
    x_updated = state_evolution.get_x_updated(vehicle_speed[i], theta[i], beta_current)
    y_updated = state_evolution.get_y_updated(vehicle_speed[i], theta[i], beta_current)
    theta_updated = state_evolution.get_theta_updated(yaw_rate_current)

    # update the system
    i = i + 1
    beta_current = beta_updated
    yaw_rate_current = yaw_rate_updated
    x_current = x_updated
    y_current = y_updated
    theta_current = theta_updated
    current_vehicle_speed = vehicle_speed_updated
    phi_current = phi_updated

j = [i, i]
time = np.arange(0, i * dt, dt)

# following is the script for exporting arrays as csv files
'''savetxt('timepd.csv', time, delimiter=',')
savetxt('xpd.csv', x, delimiter=',')
savetxt('ypd.csv', y, delimiter=',')
savetxt('phipd.csv', phi, delimiter=',')
savetxt('errorpd.csv', error, delimiter=',')
savetxt('thetapd.csv', theta_error, delimiter=',')
savetxt('X.csv', ref_X, delimiter=',')
savetxt('Y.csv', ref_Y, delimiter=',')
savetxt('ipd.csv', j, delimiter=',')'''
#####################################
## Plot vehicle behaviour
#####################################

print("Lateral ISE = {}".format(lateral_ise))
print("Heading ISE = {}".format(heading_ise))

plt.figure(1)
plt.plot(ref_X, ref_Y, label="ref path")
plt.plot(x, y, label="vehicle path")
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.legend(loc='best')
plt.axis('equal')

plt.figure(2)
plt.subplot(311)
plt.plot(time[0:i], error[0:i], linewidth=1, label=r'$d$')
plt.xlabel('Time')
plt.ylabel('lateral Error')
plt.grid(True)
plt.legend(loc='best')

plt.subplot(312)
plt.plot(time[0:i], theta_error[0:i], linewidth=1, label=r'$\theta_e$')
plt.xlabel('Time')
plt.ylabel('heading Error')
plt.grid(True)
plt.legend(loc='best')

plt.subplot(313)
plt.plot(time[0:i], phi[0:i], linewidth=1, label=r'$\Phi$', color = 'red')
plt.xlabel('Time')
plt.ylabel('Steering Angle')
plt.grid(True)
plt.legend(loc='best')

plt.figure(3)
plt.subplot(411)
plt.plot(time[0:i], path_curvature[0:i], linewidth=1, label=r'$\kappa$')
plt.xlabel('Time')
plt.ylabel("curvature")
plt.grid(True)
plt.legend(loc='best')

plt.subplot(412)
plt.plot(time[0:i], phi[0:i], linewidth=1, label=r'$\Phi$', color = 'red')
plt.plot(time[0:i], u[0:i], linewidth=1, label=r'$u$')
plt.xlabel('Time')
plt.ylabel("Angles")
plt.grid(True)
plt.legend(loc='best')

plt.subplot(413)
plt.plot(time[0:i], steering_rate[0:i], linewidth=1, label=r'$\frac{d\phi}{dt}$')
plt.xlabel('Time')
plt.ylabel("Steering Rate")
plt.grid(True)
plt.legend(loc='best')

plt.subplot(414)
plt.plot(time[0:i], vehicle_speed[0:i], linewidth=1, label=r'$v$', color='red')
plt.xlabel('Time')
plt.ylabel("speed")
plt.grid(True)
plt.legend(loc='best')

plt.show()
