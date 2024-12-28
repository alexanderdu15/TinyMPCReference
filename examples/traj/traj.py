import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from src.quadrotor import QuadrotorDynamics
from src.tinympc import TinyMPC
from utils.visualization import visualize_trajectory, plot_iterations, plot_rho_history
from utils.traj_simulation import simulate_with_controller
from scipy.spatial.transform import Rotation as spRot
from utils.reference_trajectories import Figure8Reference
from src.rho_adapter import RhoAdapter
import matplotlib.pyplot as plt

def compute_tracking_error(x_all, trajectory, dt):
    """Compute L2 tracking error over the trajectory"""
    errors = []
    times = np.arange(len(x_all)) * dt
    
    for t, x in zip(times, x_all):
        x_ref = trajectory.generate_reference(t)
        pos_error = np.linalg.norm(x[0:3] - x_ref[0:3])
        errors.append(pos_error)
    
    return np.mean(errors), np.max(errors), errors

def main(use_rho_adaptation=False):
    # Create quadrotor instance
    quad = QuadrotorDynamics()

    # Initialize hover state (for linearization)
    rg = np.array([0.0, 0, 0.0])
    qg = np.array([1.0, 0, 0, 0])
    vg = np.zeros(3)
    omgg = np.zeros(3)
    xg = np.hstack([rg, qg, vg, omgg])
    ug = quad.hover_thrust

    # Get linearized system
    A, B = quad.get_linearized_dynamics(xg, ug)

    x0 = np.copy(xg)
    
    # Trajectory parameters
    amplitude = 0.5
    w = 2*np.pi/2.5
    trajectory = Figure8Reference(A=amplitude, w=w)
    
    # Get the initial reference point and set initial state
    x_ref_0 = trajectory.generate_reference(0.0)
    x0[0:3] = x_ref_0[0:3]
    x0[3:7] = qg
    x0[7] = x_ref_0[6]
    x0[8] = 0.0
    x0[9] = x_ref_0[8]
    x0[10:13] = np.zeros(3)

    # Cost matrices
    max_dev_x = np.array([
        0.1, 0.1, 0.1,    # position
        0.5, 0.5, 0.5,    # attitude
        0.5, 0.5, 0.5,    # velocity
        0.7, 0.7, 0.2     # angular velocity
    ])
    max_dev_u = np.array([0.5, 0.5, 0.5, 0.5])
    Q = np.diag(1./max_dev_x**2)
    R = np.diag(1./max_dev_u**2)

    # Setup MPC
    N = 25
    initial_rho = 1.0
    
    # Initialize MPC
    mpc = TinyMPC(
        A=A,
        B=B,
        Q=Q,
        R=R,
        Nsteps=N,
        rho=initial_rho
    )

    # Initialize rho adapter if needed
    rho_adapter = None
    if use_rho_adaptation:
        rho_adapter = RhoAdapter(rho_base=initial_rho, rho_min=1.0, rho_max=20.0)

    # Set bounds
    u_max = [1.0-ug[0]] * quad.nu
    u_min = [-ug[0]] * quad.nu
    x_max = [5.0] * 3 + [2.0] * 9
    x_min = [-5.0] * 3 + [-2.0] * 9
    mpc.set_bounds(u_max, u_min, x_max, x_min)

    # Create trajectory reference
    x_nom = np.zeros((quad.nx, mpc.N))
    u_nom = np.zeros((quad.nu, mpc.N-1))
    for i in range(mpc.N):
        x_nom[:,i] = trajectory.generate_reference(i*quad.dt)
    u_nom[:] = ug.reshape(-1,1)

    # Run simulation
    try:
        print(f"Starting trajectory tracking simulation{'with rho adaptation' if use_rho_adaptation else ''}...")
        simulation_result = simulate_with_controller(
            x0=x0,
            x_nom=x_nom,
            u_nom=u_nom,
            mpc=mpc,
            quad=quad,
            trajectory=trajectory,
            rho_adapter=rho_adapter,
            NSIM=150 
        )

        # Unpack results based on whether we're using rho adaptation
        if use_rho_adaptation:
            x_all, u_all, iterations, rho_history = simulation_result
        else:
            x_all, u_all, iterations = simulation_result
            rho_history = None

        # Compute and display tracking error
        avg_error, max_error, errors = compute_tracking_error(x_all, trajectory, quad.dt)
        print("\nTracking Error Statistics:")
        print(f"Average L2 Error: {avg_error:.4f} meters")
        print(f"Maximum L2 Error: {max_error:.4f} meters")

        # Save data
        data_dir = Path('../data')
        (data_dir / 'iterations').mkdir(parents=True, exist_ok=True)
        np.savetxt(data_dir / 'iterations' / f"{'adaptive' if use_rho_adaptation else 'normal'}_traj.txt", iterations)
        
        if use_rho_adaptation:
            (data_dir / 'rho_history').mkdir(parents=True, exist_ok=True)
            np.savetxt(data_dir / 'rho_history' / 'adaptive_traj.txt', rho_history)

        # Visualize results
        visualize_trajectory(x_all, u_all, trajectory=trajectory, dt=quad.dt)
        plot_iterations(iterations)
        if use_rho_adaptation:
            plot_rho_history(rho_history)

        # Plot tracking error
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(len(errors))*quad.dt, errors)
        plt.xlabel('Time [s]')
        plt.ylabel('L2 Position Error [m]')
        plt.title('Tracking Error over Time')
        plt.grid(True)
        plt.show()

        print("\nSimulation completed successfully!")
        print(f"Average iterations per step: {np.mean(iterations):.2f}")
        print(f"Total iterations: {sum(iterations)}")

    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapt', action='store_true', help='Enable rho adaptation')
    args = parser.parse_args()
    
    main(use_rho_adaptation=args.adapt)