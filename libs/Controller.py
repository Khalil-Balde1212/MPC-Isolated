from abc import ABC, abstractmethod


class SISOControllers(ABC):
    def __init__(self, plant, dt=0.01, setpoint=0.0):
        self.control_history = []
        self.output_history = []
        self.control_times = []
        self.plant = plant # simpy process
        self.control_input = 0.0
        self.dt = dt
        self.setpoint = setpoint
        
        
    def update_histories(self):
        self.control_history.append(self.control_input)
        self.output_history.append(self.plant.y)
        self.control_times.append(len(self.control_history) * self.dt)


    def collect_output(self, env, duration):
        self.output_history = []
        while env.now < duration:
            self.output_history.append(self.plant.y)
            yield env.timeout(self.dt)
        return self.output_history

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def run(self, env):
        pass


# PID Controller implementation
class PIDController(SISOControllers):
    def __init__(self, model, kp=1.0, ki=0.0, kd=0.0, dt=0.01, setpoint=0.0):
        super().__init__(model)
        self.kp, self.ki, self.kd = kp, ki, kd
        self.perror, self.ierror, self.derror = 0.0, 0.0, 0.0
        self.last_error = 0.0
        self.dt = dt
        self.setpoint = setpoint

        self.max_output = 2.0
        self.min_output = 0.0

    def step(self):
        error = self.setpoint - self.plant.y
        self.ierror += error * self.dt
        self.derror = (error - self.last_error) / self.dt
        u = self.kp * error + self.ki * self.ierror + self.kd * self.derror
        # u = max(self.min_output, min(self.max_output, u))  # Clamp the output


        self.plant.set_input(u)
        self.control_input = u
        self.update_histories()
        self.last_error = error

    def run(self, env):
        while True:
            self.step()
            yield env.timeout(self.dt)
