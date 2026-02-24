import simpy
import numpy as np
 
from libs.Simulation import FirstOrderPlant
from libs.Controller import PIDController

plant_dt = 0.001
cont_dt = 0.01

def main():
    env = simpy.Environment()

    rng = np.random.default_rng(42)
    closed_plant = FirstOrderPlant(kp=1000, time_constant=0.8, dt=plant_dt, std=0, rng=rng)
    open_plant = FirstOrderPlant(kp=1000, time_constant=0.8, dt=plant_dt, std=0, rng=rng)
    controller = PIDController(closed_plant, kp=0.05, ki=0.01, kd=0.0, dt=cont_dt, setpoint=500)
    open_plant.set_input(0.5)  # Initial input for open loop

    env.process(controller.run(env))
    env.process(open_plant.run(env))
    env.process(closed_plant.run(env))

    env.run(until=5)

    import matplotlib.pyplot as plt

    # Plot closed-loop (PID) response
    plt.plot(closed_plant.times, closed_plant.output_history, label='Closed Loop (PID)')
    # Plot open-loop response
    plt.plot(open_plant.times, open_plant.output_history, label='Open Loop')
    plt.xlabel('Time (s)')
    plt.ylabel('Plant Output')
    plt.title('Open vs Closed Loop Step Response')
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == "__main__":
    main()