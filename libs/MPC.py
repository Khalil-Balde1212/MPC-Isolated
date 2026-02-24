from libs.Controller import SISOControllers
from libs.Simulation import FirstOrderPlant
import copy

from abc import abstractmethod

import numpy as np
import simpy

class DynamicMatrixController(SISOControllers):
    
    def __init__(self, plant, dt=0.01, setpoint=0.0, prediction_horizon=10, control_horizon=5, lambda_reg=None):
        super().__init__(plant, dt = dt , setpoint = setpoint)
        self.plant = plant

        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon

        self.max_control_input = 5.0
        self.min_control_input = 0.0

        # ensure some regularization by default to avoid singular matrices
        self.lambda_reg = lambda_reg if lambda_reg is not None else 1e-6

        self.reference_trajectory = np.ones(self.prediction_horizon) * setpoint
        self.control_trajectory = np.zeros(self.control_horizon)
        self.prediction_trajectory = np.zeros(self.prediction_horizon)


        # build DMC
        self.DynamicMatrix = np.zeros((self.prediction_horizon, self.control_horizon))
        plant_copy = copy.deepcopy(self.plant)
        plant_copy.set_input(1)  # Initial input for DMC construction

        dmc_constructor = simpy.Environment()
        dmc_constructor.process(plant_copy.run(dmc_constructor))
        dmc_constructor.run(until=self.prediction_horizon * self.dt)

        # Fill dynamic matrix using step response
        step_response = plant_copy.output_history
        for i in range(self.prediction_horizon):
            for j in range(self.control_horizon):
                if i >= j:
                    idx = i - j
                    if idx < len(step_response):
                        self.DynamicMatrix[i, j] = step_response[idx]
                    else:
                        self.DynamicMatrix[i, j] = step_response[-1]  # Use last value if out of bounds
                else:
                    self.DynamicMatrix[i, j] = 0.0
        print("Dynamic Matrix:\n", self.DynamicMatrix)


    def step(self):
        # Current plant output
        y0 = self.plant.y

        # Predict future outputs: y_pred = DynamicMatrix @ control_trajectory + y0
        self.prediction_trajectory = self.DynamicMatrix @ self.control_trajectory + y0

        # Compute future errors
        error_trajectory = self.reference_trajectory - self.prediction_trajectory

        # Optimization (DMC control law)
        # A = (D^T D + λI)^(-1) D^T
        A = self.DynamicMatrix.T @ self.DynamicMatrix # A= D^T D
        A += self.lambda_reg * np.eye(self.control_horizon)  # Regularization
        A = np.linalg.inv(A)  # Invert A
        A = A @ self.DynamicMatrix.T  

        delta_u = A @ error_trajectory

        # Apply control input constraints
        self.control_trajectory += delta_u
        self.control_trajectory = np.clip(self.control_trajectory, self.min_control_input, self.max_control_input)

        self.plant.set_input(self.control_trajectory[0])


        self.update_histories()

    

    def run(self, env):
        self.step()  # Initial step to set first control input
        while True:
            self.step()
            yield env.timeout(self.dt)

