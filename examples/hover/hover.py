import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from src.quadrotor import QuadrotorDynamics
from src.tinympc import TinyMPC
from utils.visualization import visualize_trajectory, plot_iterations
from utils.hover_simulation import simulate_with_controller
from scipy.spatial.transform import Rotation as spRot

def main():
    # Create quadrotor instance
    quad = QuadrotorDynamics()

    # Initialize goal state
    rg = np.array([0.0, 0, 0.0])
    qg = np.array([1.0, 0, 0, 0])
    vg = np.zeros(3)
    omgg = np.zeros(3)
    xg = np.hstack([rg, qg, vg, omgg])
    ug = quad.hover_thrust

    # Get linearized system
    A, B = quad.get_linearized_dynamics(xg, ug)

    # Initial state (offset from hover)
    x0 = np.copy(xg)
    x0[0:3] += rg + np.array([0.2, 0.2, -0.2])
    x0[3:7] = quad.rptoq(np.array([1.0, 0.0, 0.0]))

    # Cost matrices
    max_dev_x = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.05, 0.5, 0.5, 0.5, 0.7, 0.7, 0.2])
    max_dev_u = np.array([0.5, 0.5, 0.5, 0.5])/6
    Q = np.diag(1./max_dev_x**2)
    R = np.diag(1./max_dev_u**2)


    def dlqr(A, B, Q, R, n_steps=500):
        P = Q
        for i in range(n_steps):
            K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
            P = Q + A.T @ P @ (A - B @ K)
        return K, P

    K_lqr, P_lqr = dlqr(A, B, Q, R)

    

    # Setup MPC
    N = 10
    input_data = {
        'rho': 85.0,
        'A': A,
        'B': B,
        'Q': P_lqr,
        'R': R
    }

    mpc = TinyMPC(input_data, N)

    # Set bounds
    u_max = [1.0-ug[0]] * quad.nu
    u_min = [-ug[0]] * quad.nu
    x_max = [1000.] * quad.nx
    x_min = [-1000.] * quad.nx
    mpc.set_bounds(u_max, u_min, x_max, x_min)

    # Set nominal trajectory
    R0 = spRot.from_quat(qg)
    eulerg = R0.as_euler('zxy')
    xg_euler = np.hstack((eulerg, xg[4:]))
    x_nom = np.tile(0*xg_euler, (N,1)).T
    u_nom = np.tile(ug, (N-1,1)).T

    # Run simulation
    x_all, u_all, iterations = simulate_with_controller(x0, x_nom, u_nom, mpc, quad)

    # Save iterations
    np.savetxt('../data/iterations/normal_hover.txt', iterations)

    # Visualize results
    visualize_trajectory(x_all, u_all, xg, ug)
    plot_iterations(iterations)

if __name__ == "__main__":
    main()