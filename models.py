import numpy as np
import math
import collections


class Model:
  def __init__(self, speed,wheelbase):
    self.speed = speed
    self.wheelbase = wheelbase
    self.A = [[0, speed], [0, 0]] # - state matrix A
    self.B = [[0], [speed/wheelbase]] # - Control matrix B

  def set_speed(self, speed): # optional
    self.speed = speed

  def get_state_matrix(self):
    return self.A

  def get_control_matrix(self):
    return self.B


# The transfer function for the model we use for steering behaviour(first order) is: y / u = K / (1 + tau.s)
# Discretizing the model, gives: y[k+1] = -a*y[k] + b * [k]
# with a = -exp(-Ts/tau) and b = K(1 - exp(-Ts/tau))
class DiscreteFirstOrderModel:
    def __init__(self, K, tau, ts, td):
        self.a1 = -math.exp(-ts / tau)
        self.b1 =  K * (1 - math.exp(-ts / tau))
        self.ts = ts
        self.td = td
        self.y_prev = 0
        # Build buffer of commands
        # Command delay (delay (s) = ndelay * Ts)
        self.ndelay_steering = int(td / ts) + 1
        self.u_stack = collections.deque(maxlen=self.ndelay_steering)

    def get_output(self, u):
        # Apply the delay on command
        # Stack control intput
        self.u_stack.append(u)
        if (len(self.u_stack) == self.ndelay_steering):
            # If accel buffer is full, publish left value
            cmd = self.u_stack.popleft()
        else:
            cmd = 0
        y = -self.a1 * self.y_prev + self.b1 * cmd
        self.y_prev = y
        return y


class SpeedController:
    # The speed controller gives the speed of vehicle based on a fixed acceleration/deceleration. First the speed target is updated w.r.t
    # lateral error limitations and then the speed is obtained with respect to this target and current vehicle speed.
    def __init__(self, lateral_error_limit_speed, lateral_error_upper_limit, lateral_error_lower_limit, accel, speed_target, dt):
        self.lateral_error_limit_speed = lateral_error_limit_speed
        self.lateral_error_upper_limit = lateral_error_upper_limit
        self.lateral_error_lower_limit = lateral_error_lower_limit
        self.accel = accel
        self.speed_target = speed_target
        self.dt = dt
        self.speed_tolerance = accel * dt

    def get_vehicle_speed_wrt_lateral_error(self, lateral_error):
        # This function gives the speed of vehicle as a function of lateral error
        if lateral_error >= self.lateral_error_upper_limit: # low speed for high lateral error
            speed = self.lateral_error_limit_speed
        elif lateral_error <= self.lateral_error_lower_limit: # same speed for low lateral_error
            speed = self.speed_target
        else: # linear interpolation of speed between the upper and lower limits of lateral error
            speed = self.speed_target + ((self.lateral_error_limit_speed - self.speed_target) /( self.lateral_error_upper_limit - self.lateral_error_lower_limit)) * (lateral_error - self.lateral_error_lower_limit)
        return speed

    def get_vehicle_speed_wrt_acceleration(self, speed_target, current_speed):
        # This function updates the speed target based on a fixed value of acceleration/deceleration
        if current_speed > speed_target + self.speed_tolerance:
            speed = current_speed - self.accel * self.dt
        elif current_speed < speed_target - self.speed_tolerance:
            speed = current_speed + self.accel * self.dt
        else:
            speed = current_speed
        return speed


class StateEstimation:
    # In the actual vehicle we use the state prediction to calculate errors at a future state and then apply control law on them
    def __init__(self, steering_delay, b):
        self.steering_delay = steering_delay
        self.b = b

    def get_state_estimation(self, speed, steering, pose, theta):
        X = pose[0]
        Y = pose[1]
        distance = speed * self.steering_delay # Assuming constant speed
        curvature = np.tan(steering) / self.b
        heading_increment = distance * curvature
        # Prediction in local vehicle frame
        if abs(steering) > 1e-6:
            x_local_predicted = np.sin(heading_increment) / curvature
            y_local_predicted = (1 - np.cos(heading_increment)) / curvature
        else:
            x_local_predicted = distance
            y_local_predicted = 0.0
        # Rotation of local axis back to the global frame by an angle theta (heading) in the clockwise direction
        x_local_rotated = x_local_predicted * np.cos(theta) + y_local_predicted * (-np.sin(theta))
        y_local_rotated = x_local_predicted * np.sin(theta) + y_local_predicted * np.cos(theta)
        # Prediction in Global frame
        x_global_predicted = X + x_local_rotated
        y_global_predicted = Y + y_local_rotated
        predicted_pose = [x_global_predicted, y_global_predicted]
        return predicted_pose


class KinematicModel:
    def __init__(self, x_init, y_init, theta_init, dt):
        self.x_previous = x_init
        self.y_previous = y_init
        self.theta_previous = theta_init
        self.dt = dt

    def get_x_updated(self, V, theta, beta):
        x = self.x_previous + self.dt * V * np.cos(theta)
        self.x_previous = x
        return x

    def get_y_updated(self, V, theta, beta):
        y = self.y_previous + self.dt * V * np.sin(theta)
        self.y_previous = y
        return y

    def get_theta_updated(self, theta_dot):
        theta = self.theta_previous + self.dt * theta_dot
        self.theta_previous = theta
        return theta

    def get_beta_and_psi_dot(self, phi, b, V):
        beta = np.arctan(np.tan(phi) / 2.0)
        psi_dot = (V / b) * np.tan(phi)
        return beta, psi_dot



class DynamicModel:
    def __init__(self, dynamic_model_parameters, x_init, y_init, theta_init, beta_init, psi_dot_init, dt):
        self.lf = dynamic_model_parameters[0]    # distance from front axle to C.G
        self.lr = dynamic_model_parameters[1]    # distance from real axle to C.G
        self.C = dynamic_model_parameters[2]     # cornering stiffness
        self.m = dynamic_model_parameters[3]     # mass of the vehicle
        self.Iz = dynamic_model_parameters[4]    # Polar moment of inertia
        self.x_previous = x_init
        self.y_previous = y_init
        self.theta_previous = theta_init
        self.beta_previous = beta_init
        self.psi_dot_previous = psi_dot_init
        self.dt = dt

    def get_beta_and_psi_dot(self, phi, b, V):
        A00, A01, A10, A11 = self.get_matrix_A(V)
        B0, B1 = self.get_matrix_B(V)
        beta_dot = A00 * self.beta_previous + A01 * self.psi_dot_previous + B0 * phi
        psi_double_dot = A10 * self.beta_previous + A11 * self.psi_dot_previous + B1 * phi
        beta = self.beta_previous + self.dt * beta_dot
        psi_dot = self.psi_dot_previous + self.dt * psi_double_dot
        self.beta_previous = beta
        self.psi_dot_previous = psi_dot
        return beta, psi_dot

    def get_x_updated(self, V, theta, beta):
        x = self.x_previous + self.dt * V * np.cos(theta + beta)
        self.x_previous = x
        return x

    def get_y_updated(self, V, theta, beta):
        y = self.y_previous + self.dt * V * np.sin(theta + beta)
        self.y_previous = y
        return y

    def get_theta_updated(self, theta_dot):
        theta = self.theta_previous + self.dt * theta_dot
        self.theta_previous = theta
        return theta

    def get_matrix_A(self, V):
        A00 = (-2 * self.C) / (self.m * V)
        A01 = -1 + self.C * (self.lr - self.lf) / (self.m * V**2)
        A10 = self.C * (self.lr - self.lf) / self.Iz
        A11 = -self.C * (self.lr**2 + self.lf**2) / (self.Iz * V)
        return A00, A01, A10, A11

    def get_matrix_B(self, V):
        B0 = self.C / (self.m * V)
        B1 = self.lf * self.C / self.Iz
        return B0, B1
