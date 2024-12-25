# examples/hover_adapt.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from src.quadrotor import QuadrotorDynamics
from src.tinympc import TinyMPC
from src.rho_adapter import RhoAdapter
from utils.visualization import visualize_trajectory, plot_iterations
from utils.hover_simulation import simulate_with_controller
from scipy.spatial.transform import Rotation as spRot

def plot_rho_history(rho_history):
    """Plot the history of rho values"""
    plt.figure(figsize=(10, 5))
    plt.plot(rho_history, 'b.-')
    plt.xlabel('Update Step')
    plt.ylabel('Rho Value')
    plt.title('Rho Adaptation History')
    plt.grid(True)
    plt.show()

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
    x0[0:3] += np.array([0.2, 0.2, -0.2])
    x0[3:7] = quad.rptoq(np.array([1.0, 0.0, 0.0]))

    # Cost matrices
    max_dev_x = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.05, 0.5, 0.5, 0.5, 0.7, 0.7, 0.2])
    max_dev_u = np.array([0.5, 0.5, 0.5, 0.5])/6
    Q = np.diag(1./max_dev_x**2)
    R = np.diag(1./max_dev_u**2)

    # DLQR for terminal cost
    def dlqr(A, B, Q, R, n_steps=500):
        P = Q
        for i in range(n_steps):
            K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
            P = Q + A.T @ P @ (A - B @ K)
        return K, P

    K_lqr, P_lqr = dlqr(A, B, Q, R)

    # Setup MPC with adaptive rho
    N = 10
    initial_rho = 85.0
    input_data = {
        'rho': initial_rho,
        'A': A,
        'B': B,
        'Q': P_lqr,  # Using DLQR terminal cost
        'R': R
    }

    mpc = TinyMPC(input_data, N)
    rho_adapter = RhoAdapter(rho_base=initial_rho, rho_min=60.0, rho_max=100.0)

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

    try:
        # Run simulation
        print("Starting simulation with adaptive rho...")
        x_all, u_all, iterations, rho_history = simulate_with_controller(
            x0, x_nom, u_nom, mpc, quad, rho_adapter, NSIM=100
        )

        # Create data directory if it doesn't exist
        Path('../data/iterations').mkdir(parents=True, exist_ok=True)
        Path('../data/rho_history').mkdir(parents=True, exist_ok=True)

        # Save data
        np.savetxt('../data/iterations/adaptive_hover.txt', iterations)
        np.savetxt('../data/rho_history/adaptive_hover.txt', rho_history)

        # Print summary statistics
        print(f"\nSimulation completed successfully!")
        print(f"Average iterations per step: {np.mean(iterations):.2f}")
        print(f"Final rho value: {rho_history[-1]:.2f}")
        print(f"Rho range: [{min(rho_history):.2f}, {max(rho_history):.2f}]")

        # Visualize results
        visualize_trajectory(x_all, u_all, xg, ug)
        plot_iterations(iterations)
        plot_rho_history(rho_history)

    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        raise

if __name__ == "__main__":
    main()