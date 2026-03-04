import simpy
import numpy as np
from libs.hoopBead import HoopBead
from mpl_toolkits.mplot3d import Axes3D

from libs.Controller import PIDController

import matplotlib.pyplot as plt

def main():
    env = simpy.Environment()
    hoop_bead = HoopBead(radius=0.4, mass=2.0, b=-1, gravity=-9.81, dt=0.01)
    hoop_bead.reset(angle=np.deg2rad(1), angular_velocity=0.0)  # start at 1 degree with no initial velocity
    hoop_bead.set_input(5.0)

    pid_controller = PIDController(hoop_bead, kp=1.0, ki=0.0, kd=0.0, dt=0.01, setpoint=90.0)
    env.process(pid_controller.run(env))
    env.process(hoop_bead.run(env))
    env.run(until=20)

    print("Angle History:", hoop_bead.angle_history)
    print("Time History:", hoop_bead.time_history)



    fig = plt.figure()
    # Set the initial 3D view so the z axis (angular velocity) is vertical and time is horizontal
    # Elevation 20, azimuth -90 makes the view normal to the time axis
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=0, azim=0, roll =0)  # Adjust the view angle as needed
    ax.plot(
        hoop_bead.time_history,
        np.rad2deg(hoop_bead.angle_history),
        np.rad2deg(hoop_bead.angle_velocity_history),
        label='Angle & Angular Velocity'
    )
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (deg)')
    ax.set_zlabel('Angular Velocity (deg/s)')
    
    ax.set_title('3D Plot: Angle and Angular Velocity Over Time')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()

