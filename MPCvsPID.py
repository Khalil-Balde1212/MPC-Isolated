import simpy
import numpy as np  
 
from libs.Simulation import FirstOrderPlant
from libs.Controller import PIDController
from libs.MPC import DynamicMatrixController

plant_dt = 0.001
cont_dt = 0.005

def main():
    env = simpy.Environment()

    # share a random number generator so noise sequences match  
    rng = np.random.default_rng(42)  # fixed seed for reproducibility
    pid_plant = FirstOrderPlant(kp=1000, time_constant=0.3, dt=plant_dt, std=0, rng=rng)
    dmc_plant = FirstOrderPlant(kp=1000, time_constant=0.3, dt=plant_dt, std=0, rng=rng)

    # pid_controller = PIDController(pid_plant, kp=0.00596, ki=0.0284, kd=0.0, dt=cont_dt, setpoint=500)
    pid_controller = PIDController(pid_plant, kp=0.1, ki=0.005, kd=0.0, dt=cont_dt, setpoint=500)
    dmc_controller = DynamicMatrixController(dmc_plant, dt=cont_dt, setpoint=500, prediction_horizon=10, control_horizon=5, lambda_reg=1.2)
    

    # PID
    env.process(pid_controller.run(env))
    env.process(pid_plant.run(env))

    # Walk this way
    env.process(dmc_controller.run(env))
    env.process(dmc_plant.run(env))

    env.run(until=0.1)

    # Talk this way
    print("PID Output History:", [float(y) for y in pid_plant.output_history])
    print("DMC Output History:", [float(y) for y in dmc_plant.output_history])
    import matplotlib.pyplot as plt

    # Plot closed-loop (PID) response
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot closed-loop responses
    ax1.plot(pid_plant.times, pid_plant.output_history, label='Closed Loop (PID)')
    ax1.plot(dmc_plant.times, dmc_plant.output_history, label='Dynamic Matrix Control (MPC)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Plant Output')
    ax1.set_title('Open vs Closed Loop Step Response')
    ax1.legend()
    ax1.grid()
    
    # Plot control signals
    ax2.plot(pid_plant.times, pid_plant.control_history, label='PID Control Signal')
    ax2.plot(dmc_plant.times, dmc_plant.control_history, label='DMC Control Signal')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Control History')
    ax2.set_title('Control Signals')
    ax2.set_ylim(0, 5.1)
    ax2.legend()
    ax2.grid()
    plt.show()



if __name__ == "__main__":
    main()