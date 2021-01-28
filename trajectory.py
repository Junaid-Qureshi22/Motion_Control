# This script is used to define the different reference trajectories and their corresponding dynamic_parameters
# like heading, curvature etc. The cofficients for each curve are imported and then the trajectories are defined
# discretely. For ach curve there are parametric terms(Hxy or Bxy) where x represents the number and y the order
# of derivative.For example, B02 represents 2nd derivative of 1st(starts with 0) Bezier parametric term.
import numpy as np
import math
import yaml

with open("path_params.yaml") as f:
    params = yaml.safe_load(f)
line_size = params["line"]["line_size"]
hermite_size = params["hermite"]["sample_size"]
n_h = params["hermite"]["discritization_number"]
bezier_size = params["bezier"]["sample_size"]
n_b = params["bezier"]["discritization_number"]
loop1_size = params["loop1"]["sample_size"]
n_l1 = params["loop1"]["discritization_number"]
loop2_size = params["loop2"]["sample_size"]
n_l2 = params["loop2"]["discritization_number"]



class LinePath:
    def get_path_information(self):
        x = np.linspace(0, 100, line_size)
        y = x
        C = np.array(list(zip(x, y)))
        y1 = np.ones(line_size)
        curvature = np.zeros(line_size)
        return C, x, y, curvature, np.arctan(y1), line_size


class HermitePath:
    def __init__(self, num, sign_coeff):
        self.num = num                  # To select number of hermite curve e.g: hermite1 or hermite2
        self.sign_coeff = sign_coeff    # To select right or left turn
        if self.num == 1:
            file = "hermite1.yaml"
        elif self.num == 2:
            file = "hermite2.yaml"
        with open(file) as f:
            params = yaml.safe_load(f)
        # coefficients of hermite curves
        self.p0x = params["coefficients"]["p0x"]
        self.p0y = params["coefficients"]["p0y"]
        self.v0x = params["coefficients"]["v0x"]
        self.v0y = params["coefficients"]["v0y"]
        self.p1x = params["coefficients"]["p1x"]
        self.p1y = params["coefficients"]["p1y"]
        self.v1x = params["coefficients"]["v1x"]
        self.v1y = params["coefficients"]["v1y"]


    def get_path_information(self):
        x = np.zeros(hermite_size)
        y = np.zeros(hermite_size)
        theta = np.zeros(hermite_size)
        curvature = np.zeros(hermite_size)
        for i in range(hermite_size):
            t = i/n_h                     # parameter t
            H00 = 2 * t**3 - 3 * t**2 + 1 # parametric term1
            H10 = t**3 - 2 * t**2 + t     # parametric term2
            H20 = t**3 - t**2             # parametric term3
            H30 = 3 * t**2 - 2 * t**3     # parametric term4
            H01 = 6 * t * (t - 1)         # 1st derivative of parametric term1
            H11 = 3 * t**2 - 4 * t + 1    # 1st derivative of parametric term2
            H21 = 3 * t**2 - 2 * t        # 1st derivative of parametric term3
            H31 = 6 * t * (1 - t)         # 1st derivative of parametric term4
            H02 = 6 * (2 * t - 1)         # 2nd derivative of parametric term1
            H12 = 6 * t - 4               # 2nd derivative of parametric term2
            H22 = 6 * t - 2               # 2nd derivative of parametric term3
            H32 = 6 * (1 - 2 * t)         # 2nd derivative of parametric term4
            x[i] = (self.sign_coeff)*(H00 * self.p0x + H10 * self.v0x + H20 * self.v1x + H30 * self.p1x)  # x cordinate of curve
            y[i] = H00 * self.p0y + H10 * self.v0y + H20 * self.v1y + H30 * self.p1y                      # y cordinate of curve
            dx = (self.sign_coeff) * (H01 * self.p0x  + H11 * self.v0x + H21 * self.v1x + H31 * self.p1x) # 1st derivative of x w.r.t t
            dy = H01 * self.p0y  + H11 * self.v0y + H21 * self.v1y + H31 * self.p1y                       # 1st derivative of y w.r.t t
            d2x = (self.sign_coeff) * (H02 * self.p0x + H12 * self.v0x + H22 * self.v1x + H32 * self.p1x) # 2nd derivative of x w.r.t t
            d2y = H02 * self.p0y + H12 * self.v0y + H22 * self.v1y + H32 * self.p1y                       # 2nd derivative of y w.r.t t
            y2 = (dx * d2y - dy * d2x)/pow(dx, 3)                                                         # 2nd derivative of y w.r.t x
            curvature[i] = y2 / pow((1 + (dy/dx)**2), 1.5)                                                # curvature of path
            theta[i] = math.atan2(dy, dx)                                                                 # path heading angle
        C = np.array(list(zip(x, y)))
        return C, x, y, curvature, theta, hermite_size


class BezierCurve:
    def __init__(self, deg):
        self.deg = deg                                      # degree of Bezier curve
        if self.deg == 2:
            file = "bezier_quadratic_coefficients.yaml"
        elif self.deg == 3:
            file = "bezier_cubic_coefficients.yaml"
        with open(file) as f:
            params = yaml.safe_load(f)
        # coefficients of Bezier curves
        self.x0 = params["coefficients"]["x0"]
        self.x1 = params["coefficients"]["x1"]
        self.x2 = params["coefficients"]["x2"]
        self.x3 = params["coefficients"]["x3"]
        self.y0 = params["coefficients"]["y0"]
        self.y1 = params["coefficients"]["y1"]
        self.y2 = params["coefficients"]["y2"]
        self.y3 = params["coefficients"]["y3"]


    def get_path_information(self):
        x = np.zeros(bezier_size)
        y = np.zeros(bezier_size)
        theta = np.zeros(bezier_size)
        curvature = np.zeros(bezier_size)
        for i in range(bezier_size):
            t = i/n_b
            if self.deg == 2:
                B00 = (1-t)**2
                B10 = 2 * (1-t) * t
                B20 = t**2
                B01 = 2 * (1 - t)
                B11 = 2 * t
                x[i] = self.x0 * B00 + self.x1 * B10 + self.x2 * B20
                y[i] = self.y0 * B00 + self.y1 * B10 + self.y2 * B20
                dx = (B01 * (self.x1 - self.x0) + B11 * (self.x2 - self.x1))
                dy = (B01 * (self.y1 - self.y0) + B11 * (self.y2 - self.y1))
                d2x = ( 2 * (self.x2 - 2 * self.x1 + self.x0))
                d2y = ( 2 * (self.y2 - 2 * self.y1 + self.y0))
                y2 = (dx * d2y - dy * d2x) / pow(dx, 3)
                curvature[i] = y2 / pow((1 + (dy/dx)**2), 1.5)
                theta[i] = math.atan2(dy, dx)
            elif self.deg == 3:
                B00 = (1-t)**3
                B10 = 3 * (1-t)**2 * t
                B20 = 3 * (1-t) *t**2
                B30 = t**3
                B01 = 3*(1-t)**2
                B11 = 6*(1-t)*t
                B21 = 3*t**2
                B02 = 6 * (1 - t)
                B12 = 6 * t
                x[i] = self.x0* B00 +  self.x1 * B10 + self.x2 * B20  + self.x3 * B30
                y[i] = self.y0* B00 +  self.y1 * B10 + self.y2 * B20 + self.y3 * B30
                dx = B01 * (self.x1 - self.x0) + B11 * (self.x2 - self.x1) + B21 * (self.x3 - self.x2)
                dy = B01 * (self.y1 - self.y0) + B11 * (self.y2 - self.y1) + B21 * (self.y3 - self.y2)
                d2x = B02 * (self.x2 - 2 * self.x1 + self.x0) + 6 * t * (self.x3 - 2 * self.x2 + self.x1)
                d2y = B02 * (self.y2 - 2 * self.y1 + self.y0) + 6 * t * (self.y3 - 2 * self.y2 + self.y1)
                y2 = (dx * d2y - dy * d2x) / pow(dx, 3)
                curvature[i] = y2 / pow((1 + (dy/dx)**2), 1.5)
                theta[i] = math.atan2(dy, dx)
            C = np.array(list(zip(x, y)))
        return C, x, y, curvature, theta, bezier_size


class Loop1:
    def __init__(self):
        with open("loop1_coefficients.yaml") as f:
            params = yaml.safe_load(f)
        # coefficients of Loop1
        self.x0f = params["forward_coefficients"]["x0"]
        self.x1f = params["forward_coefficients"]["x1"]
        self.x2f = params["forward_coefficients"]["x2"]
        self.y0f = params["forward_coefficients"]["y0"]
        self.y1f = params["forward_coefficients"]["y1"]
        self.y2f = params["forward_coefficients"]["y2"]
        self.x0r = params["reverse_coefficients"]["x0"]
        self.x1r = params["reverse_coefficients"]["x1"]
        self.x2r = params["reverse_coefficients"]["x2"]
        self.y0r = params["reverse_coefficients"]["y0"]
        self.y1r = params["reverse_coefficients"]["y1"]
        self.y2r = params["reverse_coefficients"]["y2"]



    def get_path_information(self):
        x = np.zeros(loop1_size)
        y = np.zeros(loop1_size)
        theta = np.zeros(loop1_size)
        curvature = np.zeros(loop1_size)
        for i in range(loop1_size / 2): # upper half of loop 1
            t = i/n_l1
            B00 = (1-t)**2
            B10 = 2 * (1-t) * t
            B20 = t**2
            B01 = 2 * (1 - t)
            B11 = 2 * t
            x[i] = self.x0f * B00 + self.x1f * B10 + self.x2f * B20
            y[i] = self.y0f * B00 + self.y1f * B10 + self.y2f * B20
            dx = (B01 * (self.x1f - self.x0f) + B11 * (self.x2f - self.x1f))
            dy = (B01 * (self.y1f - self.y0f) + B11 * (self.y2f - self.y1f))
            d2x = ( 2 * (self.x2f - 2 * self.x1f + self.x0f))
            d2y = ( 2 * (self.y2f - 2 * self.y1f + self.y0f))
            y2 = (dx * d2y - dy * d2x) / pow(dx, 3)
            theta[i] = math.atan2(dy, dx)
            curvature[i] = y2 / pow((1 + (dy/dx)**2), 1.5)
        for i in range(loop1_size / 2, loop1_size): # lower half of loop1
            t = (i - loop1_size / 2) / n_l1
            B00 = (1-t)**2
            B10 = 2 * (1-t) * t
            B20 = t**2
            B01 = 2 * (1 - t)
            B11 = 2 * t
            x[i] = self.x0r * B00 + self.x1r * B10 + self.x2r * B20
            y[i] = self.y0r * B00 + self.y1r * B10 + self.y2r * B20
            dx = (B01 * (self.x1r - self.x0r) + B11 * (self.x2r - self.x1r))
            dy = (B01 * (self.y1r - self.y0r) + B11 * (self.y2r - self.y1r))
            d2x = ( 2 * (self.x2r - 2 * self.x1r + self.x0r))
            d2y = ( 2 * (self.y2r - 2 * self.y1r + self.y0r))
            y2 = (dx * d2y - dy * d2x) / pow(dx, 3)
            theta[i] = np.arctan2(dy, dx)
            curvature[i] = y2 / pow((1 + (dy/dx)**2), 1.5)
        C = np.array(list(zip(x, y)))
        return C, x, y, curvature, theta, loop1_size


class Loop2:
    def __init__(self):
        with open("loop2_coefficients.yaml") as f:
            params = yaml.safe_load(f)
        # coefficients of Loop2
        self.x0 = params["coefficients"]["x0"]
        self.x1 = params["coefficients"]["x1"]
        self.x2 = params["coefficients"]["x2"]
        self.x3 = params["coefficients"]["x3"]
        self.y0 = params["coefficients"]["y0"]
        self.y1 = params["coefficients"]["y1"]
        self.y2 = params["coefficients"]["y2"]
        self.y3 = params["coefficients"]["y3"]


    def get_path_information(self):
        x = np.zeros(loop2_size)
        y = np.zeros(loop2_size)
        theta = np.zeros(loop2_size)
        curvature = np.zeros(loop2_size)
        for i in range(loop2_size):
            t = i/n_l2
            B00 = (1-t)**3
            B10 = 3 * (1-t)**2 * t
            B20 = 3 * (1-t) *t**2
            B30 = t**3
            B01 = 3*(1-t)**2
            B11 = 6*(1-t)*t
            B21 = 3*t**2
            B02 = 6 * (1 - t)
            B12 = 6 * t
            x[i] = self.x0* B00 +  self.x1 * B10 + self.x2 * B20  + self.x3 * B30
            y[i] = self.y0* B00 +  self.y1 * B10 + self.y2 * B20 + self.y3 * B30
            dx = B01 * (self.x1 - self.x0) + B11 * (self.x2 - self.x1) + B21 * (self.x3 - self.x2)
            dy = B01 * (self.y1 - self.y0) + B11 * (self.y2 - self.y1) + B21 * (self.y3 - self.y2)
            d2x = B02 * (self.x2 - 2 * self.x1 + self.x0) + 6 * t * (self.x3 - 2 * self.x2 + self.x1)
            d2y = B02 * (self.y2 - 2 * self.y1 + self.y0) + 6 * t * (self.y3 - 2 * self.y2 + self.y1)
            y2 = (dx * d2y - dy * d2x) / pow(dx, 3)
            curvature[i] = y2 / pow((1 + (dy/dx)**2), 1.5)
            theta[i] = math.atan2(dy, dx)
        C = np.array(list(zip(x, y)))
        return C, x, y, curvature, theta, loop2_size
