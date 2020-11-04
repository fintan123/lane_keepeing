import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class PidController:
    #Kp: poportional gain
    #Kd: derivative gain
    #Ki: integral gain
    #Ts: Sampling time

    def __init__(self, kp, ki, kd, ts):
        """
        Constructor for PIDcontroller
        """
        self.__kp = kp
        self.__kd = kd / ts  # discrete-time Kd
        self.__ki = ki * ts
        self.__previous_error = None
        self.__error_sum = 0.

    def control(self, y, set_point=0.):

        error = set_point - y  # compute the control error
        steering_action = self.__kp * error  # P controller

        # D component:
        if self.__previous_error is not None:
            error_diff = error - self.__previous_error
            steering_action += self.__kd * error_diff

        # I component:
        # TODO: Do this as an exercise. Introduce the I component
        # here (don't forget to update the sum of errors).

        steering_action += self.__ki * self.__error_sum
        self.__error_sum += error

        self.__previous_error = error
        return steering_action


class Car:

    def __init__(self,
                 length=2.3,
                 velocity=5,
                 x_pos_init=0,
                 y_pos_init=0, # set initial y postion
                 pose_init= 0): # set initial angle for steering
        self.__length = length
        self.__velocity = velocity
        self.__x = x_pos_init
        self.__y = y_pos_init
        self.__pose = pose_init

    def move(self, steering_angle, dt):
        # This method computes the position and orientation (pose)
        # of the car after time `dt` starting from its current
        # position and orientation by solving an IVP

        def bicycle_model(_t, z):
            x = z[0]
            y = z[1]
            theta = z[2]
            return [self.__velocity * np.cos(theta),
                    self.__velocity * np.sin(theta),
                    self.__velocity * np.tan(steering_angle)
                    / self.__length]

        sol = solve_ivp(bicycle_model,
                        [0, dt],
                        [self.__x, self.__y, self.__pose])

        self.__x = sol.y[0, -1]
        self.__y = sol.y[1, -1]
        self.__pose = sol.y[2, -1]

    def y(self):
        return self.__y

    def x(self):
        return self.__x

    def pose(self):
        return self.__pose
#-----------------------------------------------

t_sampling = 0.025
car = Car(y_pos_init=0.3, pose_init= 5 * np.pi / 180)
pid = PidController(kp=0.35, kd=0.2, ki=0.02, ts=t_sampling) # kp 0.35 kd 0.2 ki 0.01

w = 1 * np.pi / 180 #constant steering angle

n_sim_points = 2000

x_cache = np.array([car.x()], dtype=float)
pose_cache = np.array([car.pose()], dtype=float)
y_cache = np.array([car.y()], dtype=float)

for i in range(n_sim_points):
    u = pid.control(car.y())
    car.move(u + w, t_sampling)
    y_cache = np.append(y_cache, car.y())
    x_cache = np.append(x_cache, car.x())
    pose_cache = np.append(pose_cache, car.pose())

t_span = t_sampling * np.arange(n_sim_points+1)
plt.plot(x_cache, y_cache)
plt.xlabel("Distance Travelled/cm")
plt.ylabel("Distance from Setpoint/cm")
plt.grid()
plt.show()
