from libs.Controller import SISOControllers
import copy

import numpy as np
import simpy

class DynamicMatrixController(SISOControllers):
    
    def __init__(self, plant, dt=0.01, setpoint=0.0, prediction_horizon=10, control_horizon=5, lambda_reg=None):
        super().__init__(plant, dt = dt , setpoint = setpoint)
        self.plant = plant

        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        self.n_outputs = len(self.plant.y)
        self.n_inputs = len(self.plant.u)

        # ensure some regularization by default to avoid singular matrices
        self.lambda_reg = lambda_reg if lambda_reg is not None else 1e-6

        self.output_keys = list(self.plant.y.keys())
        self.input_keys = list(self.plant.u.keys())
        self.input_mins, self.input_maxs = self._infer_input_bounds()

        self.reference_trajectory = []
        self.control_trajectory = []
        self.prediction_trajectory = []

        setpoint_arr = np.atleast_1d(setpoint).astype(float)
        if setpoint_arr.size == 1:
            setpoint_arr = np.repeat(setpoint_arr, self.n_outputs)
        if setpoint_arr.size != self.n_outputs:
            raise ValueError(f"setpoint must have {self.n_outputs} values")
        self.setpoint = setpoint_arr

        for i in range(self.n_outputs):
            self.reference_trajectory.append(np.ones(self.prediction_horizon) * self.setpoint[i])
            self.control_trajectory.append(np.zeros(self.control_horizon))
            self.prediction_trajectory.append(np.zeros(self.prediction_horizon))

        # Build DMC as matrix of step-response blocks
        # DynamicMatrix[i][j] -> (prediction_horizon x control_horizon)
        self.DynamicMatrix = [
            [np.zeros((self.prediction_horizon, self.control_horizon)) for _ in range(self.n_inputs)]
            for _ in range(self.n_outputs)
        ]

        step_responses = self._build_step_responses()

        self.Kss = np.array([
            [step_responses[i][j][-1] for j in range(self.n_inputs)]
            for i in range(self.n_outputs)
        ])

        try:
            u_ss, *_ = np.linalg.lstsq(self.Kss, self.setpoint, rcond=None)
            if np.any(u_ss < self.input_mins) or np.any(u_ss > self.input_maxs):
                print("Warning: requested setpoint may be infeasible with current input limits.")
        except np.linalg.LinAlgError:
            pass

        for output_index in range(self.n_outputs):
            for input_index in range(self.n_inputs):
                s = step_responses[output_index][input_index]
                block = np.zeros((self.prediction_horizon, self.control_horizon))
                for p in range(self.prediction_horizon):
                    for c in range(self.control_horizon):
                        if p >= c:
                            idx = p - c
                            block[p, c] = s[idx]
                self.DynamicMatrix[output_index][input_index] = block
        
        
        print("Dynamic Matrix block shapes:")
        for row in self.DynamicMatrix:
            print([block.shape for block in row])



    def _set_inputs_from_vector(self, plant_obj, u_vec):
        plant_obj.set_input(*[float(value) for value in u_vec])

    def _infer_input_bounds(self):
        input_mins = np.full(self.n_inputs, -np.inf)
        input_maxs = np.full(self.n_inputs, np.inf)

        if hasattr(self.plant, "G"):
            for input_index in range(self.n_inputs):
                candidate = self.plant.G[0][input_index]
                input_mins[input_index] = getattr(candidate, "min_input", -np.inf)
                input_maxs[input_index] = getattr(candidate, "max_input", np.inf)

        return input_mins, input_maxs


    def _build_step_responses(self):
        step_responses = [
            [np.zeros(self.prediction_horizon) for _ in range(self.n_inputs)]
            for _ in range(self.n_outputs)
        ]

        for input_index in range(self.n_inputs):
            plant_copy = copy.deepcopy(self.plant)

            # Disable noise on all SISO elements if present
            if hasattr(plant_copy, "G"):
                for row in plant_copy.G:
                    for plant_element in row:
                        if hasattr(plant_element, "std"):
                            plant_element.std = 0.0

            if hasattr(plant_copy, "reset"):
                plant_copy.reset()

            u_step = np.zeros(self.n_inputs)
            u_step[input_index] = 1.0
            self._set_inputs_from_vector(plant_copy, u_step)

            env = simpy.Environment()
            env.process(plant_copy.run(env))
            env.run(until=self.prediction_horizon * self.dt)

            output_hist = np.asarray(plant_copy.output_history, dtype=float)
            if output_hist.ndim == 1:
                output_hist = output_hist.reshape(-1, 1)

            for output_index in range(self.n_outputs):
                s = np.zeros(self.prediction_horizon)
                n_available = min(self.prediction_horizon, output_hist.shape[0])
                s[:n_available] = output_hist[:n_available, output_index]
                if n_available > 0 and n_available < self.prediction_horizon:
                    s[n_available:] = s[n_available - 1]
                step_responses[output_index][input_index] = s

        return step_responses


    def step(self):
        # Current plant output
        if isinstance(self.plant.y, dict):
            y0 = np.array([self.plant.y[key] for key in self.output_keys], dtype=float)
        else:
            y0 = np.atleast_1d(self.plant.y).astype(float)
            if y0.size == 1:
                y0 = np.repeat(y0, self.n_outputs)

        # Predict future outputs for each output channel:
        # y_i = sum_j D_ij @ u_j + y0_i
        for i in range(self.n_outputs):
            y_pred_i = np.zeros(self.prediction_horizon)
            for j in range(self.n_inputs):
                y_pred_i += self.DynamicMatrix[i][j] @ self.control_trajectory[j]
            self.prediction_trajectory[i] = y_pred_i + y0[i]

        # Compute future errors
        error_trajectory = [
            self.reference_trajectory[i] - self.prediction_trajectory[i]
            for i in range(self.n_outputs)
        ]
        e = np.concatenate(error_trajectory)

        # Build full block dynamic matrix D with shape (n_outputs*Np, n_inputs*Nc)
        D = np.block(self.DynamicMatrix)

        # Optimization (DMC control law)
        # A = (D^T D + λI)^(-1) * D^T
        A = D.T @ D
        A += self.lambda_reg * np.eye(self.n_inputs * self.control_horizon)
        delta_u = np.linalg.solve(A, D.T @ e)

        # Apply control input constraints
        for j in range(self.n_inputs):
            start = j * self.control_horizon
            stop = (j + 1) * self.control_horizon
            self.control_trajectory[j] += delta_u[start:stop]
            self.control_trajectory[j] = np.clip(
                self.control_trajectory[j],
                self.input_mins[j],
                self.input_maxs[j],
            )

        # Apply first move only for each input channel
        u0 = np.array([self.control_trajectory[j][0] for j in range(self.n_inputs)])
        self._set_inputs_from_vector(self.plant, u0)

        # Receding horizon: shift move plan one step ahead
        for j in range(self.n_inputs):
            if self.control_horizon > 1:
                self.control_trajectory[j][:-1] = self.control_trajectory[j][1:]
                self.control_trajectory[j][-1] = self.control_trajectory[j][-2]

        self.control_input = u0


        self.update_histories()

    

    def run(self, env):
        while True:
            self.step()
            yield env.timeout(self.dt)

