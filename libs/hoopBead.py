import simpy
import numpy as np


class HoopBead:
    def __init__(self, radius=1.0, mass=1.0, b=0.1, gravity=9.81, dt=0.01):
        self.radius = radius
        self.mass = mass
        self.b = b
        self.gravity = gravity
        self.dt = dt

        self.y = 0.0  # initial angle (radians)
        self.angular_velocity = 0.0  # initial angular velocity
        self.angular_acceleration = 0.0

        self.omega = 0.0  # control input: hoop angular velocity

        self.angle_history = []
        self.angle_velocity_history = []
        self.angle_acceleration_history = []
        self.omega_history = []
        self.time_history = []

    def step(self):
        # theta.. = omega^2 * sin cos + g/r sin + b/m *theta.
        self.angular_acceleration = (self.omega ** 2) * np.sin(self.y) * np.cos(self.y)
        self.angular_acceleration += (self.gravity / self.radius) * np.sin(self.y)
        self.angular_acceleration += (self.b / self.mass) * self.angular_velocity

        self.angular_velocity += self.angular_acceleration * self.dt
        self.y += self.angular_velocity * self.dt


        # update controller history
        self.omega_history.append(self.omega)
        self.angle_history.append(self.y)
        self.angle_velocity_history.append(self.angular_velocity)
        self.angle_acceleration_history.append(self.angular_acceleration)
        self.time_history.append(len(self.angle_history) * self.dt)

    def run(self, env):
        while True:
            self.step()
            yield env.timeout(self.dt)


    def reset(self, angle=0.0, angular_velocity=0.0):
        self.y = angle
        self.angular_velocity = angular_velocity
        self.angular_acceleration = 0.0
        self.angle_history = []
        self.time_history = []

    def deg_to_rad(self, degrees):
        return degrees * np.pi / 180.0

    def set_input(self, omega):
        self.omega = omega

    
