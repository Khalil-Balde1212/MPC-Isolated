import simpy
import numpy as np

from libs.Simulation import FirstOrderPlant



class StirredTankReactor:
    def __init__(self, dt=0.001):
        self.dt = dt
        self.y = {'concentration': 0.0, 'temperature': 0.0}
        self.u = {'feed_flow': 0.0, 'coolant_flow': 0.0}

        self.G = np.array([
            [FirstOrderPlant(kp=1, time_constant=0.7, dt=dt), FirstOrderPlant(kp=5, time_constant=0.3, dt=dt)],
            [FirstOrderPlant(kp=1, time_constant=0.5, dt=dt), FirstOrderPlant(kp=2, time_constant=0.4, dt=dt)]
        ])


        self.control_history = []
        self.output_history = []
        self.times = []


    def set_input(self, feed_flow, coolant_flow):
        self.u['feed_flow'] = feed_flow
        self.u['coolant_flow'] = coolant_flow
    
    def step(self):
        # Placeholder for actual reactor dynamics

        '''
         [Y_1]   [1/(1+0.7s)][5/0.3s] [U_1]
         [Y_2] = [1/1+0.5s][2/1+0.4s]@[U_2]
        '''
        self.G[0][0].set_input(self.u['feed_flow'])
        self.G[1][0].set_input(self.u['feed_flow'])
        self.G[0][1].set_input(self.u['coolant_flow'])
        self.G[1][1].set_input(self.u['coolant_flow'])

        for row in self.G:
            for plant in row:
                plant.step()

        self.y['concentration'] = self.G[0][0].y + self.G[0][1].y
        self.y['temperature'] = self.G[1][0].y + self.G[1][1].y



        
        # Record output and time history
        self.output_history.append((self.y['concentration'], self.y['temperature']))
        self.control_history.append((self.u['feed_flow'], self.u['coolant_flow']))
        self.times.append(len(self.output_history) * self.dt)

        return self.y

    def run(self, env):
        while True:
            self.step()
            yield env.timeout(self.dt)

    def reset(self):
        self.y = {'concentration': 0.0, 'temperature': 0.0}
        self.u = {'feed_flow': 0.0, 'coolant_flow': 0.0}
        self.output_history = []
        self.control_history = []
        self.times = []
        for row in self.G:
            for plant in row:
                plant.reset()


if __name__ == "__main__":
    env = simpy.Environment()
    plant = StirredTankReactor(dt=0.01)
    plant.set_input(1.0, 1.0)  # Step input for testing
    env.process(plant.run(env))
    env.run(until=5)

    import matplotlib.pyplot as plt
    plt.plot(plant.times, [output[0] for output in plant.output_history], label='Concentration')
    plt.plot(plant.times, [output[1] for output in plant.output_history], label='Temperature')    
    plt.xlabel('Time (s)')
    plt.ylabel('Output')
    plt.title('Stirred Tank Reactor Output')
    plt.grid()
    plt.legend()
    plt.show()


    print("Output History:", plant.output_history)
    print("Control History:", plant.control_history)