import simpy
import numpy as np  
 
from libs.Simulation import FirstOrderPlant
from libs.Controller import PIDController
from libs.MPC import DynamicMatrixController

plant_dt = 0.001
cont_dt = 0.001

def main():
    env = simpy.Environment()

    # share a random number generator so noise sequences match  
    rng = np.random.default_rng(42)  # fixed seed for reproducibility
    pid_plant = FirstOrderPlant(kp=1000, time_constant=0.8, dt=plant_dt, std=0, rng=rng)
    dmc_plant = FirstOrderPlant(kp=1000, time_constant=0.8, dt=plant_dt, std=0, rng=rng)
    # noise already configured via constructor; seed ensures identical samples
    pid_plant.std = 0
    dmc_plant.std = 0


    pid_controller = PIDController(pid_plant, kp=0.1, ki=0.001, kd=0.0, dt=cont_dt, setpoint=500)
    dmc_controller = DynamicMatrixController(dmc_plant, dt=cont_dt, setpoint=500, prediction_horizon=3, control_horizon=2)
    

    # PID
    env.process(pid_controller.run(env))
    env.process(pid_plant.run(env))

    # Walk this way
    env.process(dmc_controller.run(env))
    env.process(dmc_plant.run(env))

    env.run(until=0.25)

    # Talk this way
    print("PID Output History:", [float(y) for y in pid_plant.output_history])
    print("DMC Output History:", [float(y) for y in dmc_plant.output_history])
    import matplotlib.pyplot as plt

    # Plot closed-loop (PID) response
    plt.plot(pid_plant.times, pid_plant.output_history, label='Closed Loop (PID)')
    # Plot open-loop response
    plt.plot(dmc_plant.times, dmc_plant.output_history, label='Dynamic Matrix Control (MPC)')
    plt.xlabel('Time (s)')
    plt.ylabel('Plant Output')
    plt.title('Open vs Closed Loop Step Response')
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == "__main__":
    main()