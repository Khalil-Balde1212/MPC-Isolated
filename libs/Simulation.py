import simpy
import numpy as np

class FirstOrderPlant:
    def __init__(self, kp=1000.0, time_constant=0.1, dt=0.01, std=0.0, rng=None):
        # kp is the plant gain: steady-state output = kp * input
        # default kp=1000 makes a unit input (u=1) drive the output to 1000.
        self.kp = kp
        self.tau = time_constant
        self.dt = dt
        self.y = 0.0  # initial state
        self.u = 0.0  # control input

        self.output_history = [] 
        self.control_history = []
        self.times = []

        # non linearities
        self.max_input = 5.0
        self.min_input = 0.0

        # Noise
        self.std = std 
        # random number generator (allows two plants to share same noise)
        self.rng = rng if rng is not None else np.random

    def set_input(self, u):
        # clamp input value
        u_clamped = max(self.min_input, min(self.max_input, u))
        self.u = u_clamped

    def step(self):
        # Simple linear plant: tau*y' + y = kp*u
        self.y = self.y + (self.dt / self.tau) * (self.kp * self.u - self.y)

        # Add Gaussian noise
        if self.std > 0:
            # use assigned RNG so multiple plants can share the same sequence
            self.y += self.rng.normal(0, self.std)  

        # Record output and time history
        self.output_history.append(self.y)
        self.control_history.append(self.u)
        self.times.append(len(self.output_history) * self.dt)
        return self.y
    
    def run(self, env):
        while True:
            self.step()
            yield env.timeout(self.dt)

    def reset(self):
        self.y = 0.0
        self.u = 0.0
        self.output_history = []
        self.control_history = []
        self.times = []

    def get_steady_state(self):
        return self.kp * self.u