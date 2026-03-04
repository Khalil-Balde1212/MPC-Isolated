from libs.MIMOPlant import StirredTankReactor
from libs.MIMODMC import DynamicMatrixController

import simpy
import numpy as np


if __name__ == "__main__":
    env = simpy.Environment()

    plant = StirredTankReactor(dt=0.01)
    # plant.set_input(1.0, 2.0)  # Step input for testing
    env.process(plant.run(env))

    controller = DynamicMatrixController(plant, dt=0.01, setpoint=np.array([1.5, 1.0]), prediction_horizon=20, control_horizon=5, lambda_reg=1)
    env.process(controller.run(env))

    env.run(until=2)

    # print("Plant Output History:", plant.output_history)

    import matplotlib.pyplot as plt
    # Extract concentration and temperature from output history
    concentrations = [output[0] for output in plant.output_history]
    temperatures = [output[1] for output in plant.output_history]
    
    # Plot closed-loop response
    plt.plot(plant.times, concentrations, label='Concentration')
    plt.plot(plant.times, temperatures, label='Temperature')
    plt.xlabel('Time (s)') 
    plt.ylabel('Output')
    plt.grid()
    plt.title('Closed-Loop Response of MIMO DMC on Stirred Tank Reactor')
    plt.legend()
    plt.show()

    print("Final concentration:", concentrations[-1])
    print("Final temperature:", temperatures[-1])