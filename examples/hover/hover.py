import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from src.quadrotor import QuadrotorDynamics
from src.tinympc import TinyMPC
from src.rho_adapter import RhoAdapter
from utils.visualization import visualize_trajectory, plot_iterations, plot_rho_history
from utils.hover_simulation import simulate_with_controller
from scipy.spatial.transform import Rotation as spRot
import matplotlib.pyplot as plt

def compute_hover_error(x_all, xg):
    """Compute L2 tracking error over the hover trajectory"""
    errors = []
    for x in x_all:
        pos_error = np.linalg.norm(x[0:3] - xg[0:3])
        errors.append(pos_error)
    return np.mean(errors), np.max(errors), errors

def main(use_rho_adaptation=False, use_recaching=False, use_wind=False):
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



    # Setup MPC
    N = 10
    initial_rho = getattr(main, 'last_rho', 85.0)  # Default 85.0 if first run
    
    if use_rho_adaptation:
        print(f"Using warm-started rho: {initial_rho}")
        rho_adapter = RhoAdapter(rho_base=initial_rho, rho_min=60.0, rho_max=100.0)
    else:
        rho_adapter = None
    
    # Initialize MPC with rho adapter
    mpc = TinyMPC(
        A=A,
        B=B,
        Q=Q,
        R=R,
        Nsteps=N,
        rho=initial_rho,
        rho_adapter=rho_adapter,
        recache=use_recaching
    )

    if use_rho_adaptation:
        mpc.rho_adapter.initialize_derivatives(mpc.cache)

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
        print(f"Starting hover simulation with:")
        print(f"- Rho adaptation: {'enabled' if use_rho_adaptation else 'disabled'}")
        print(f"- Cache recomputation: {'enabled' if use_recaching else 'disabled'}")
        print(f"- Wind disturbance: {'enabled' if use_wind else 'disabled'}")

        simulation_result = simulate_with_controller(
            x0=x0,
            x_nom=x_nom,
            u_nom=u_nom,
            mpc=mpc,
            quad=quad,
            NSIM=150,
            use_wind=use_wind
        )

        # Unpack results based on whether we're using rho adaptation
        if use_rho_adaptation:
            x_all, u_all, iterations, rho_history = simulation_result
        else:
            x_all, u_all, iterations = simulation_result
            rho_history = None

        # Compute tracking error
        avg_error, max_error, errors = compute_hover_error(x_all, xg)
        print("\nTracking Error Statistics:")
        print(f"Average L2 Error: {avg_error:.4f} meters")
        print(f"Maximum L2 Error: {max_error:.4f} meters")

        # Save data
        data_dir = Path('../data')
   
        suffix = '_normal'
        if use_rho_adaptation:
            suffix = '_adaptive'
        if use_wind:
            suffix += '_wind'
        if use_recaching:
            suffix += '_recache'

        np.savetxt(data_dir / 'iterations' / f"hover{suffix}.txt", iterations)
        
        if use_rho_adaptation:
            np.savetxt(data_dir / 'rho_history' / f"hover{suffix}.txt", rho_history)

        # Visualize results
        visualize_trajectory(x_all, u_all, xg, ug)
        plot_iterations(iterations)
        if use_rho_adaptation:
            plot_rho_history(rho_history)

        # Plot tracking error
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(len(errors))*quad.dt, errors)
        plt.xlabel('Time [s]')
        plt.ylabel('L2 Position Error [m]')
        plt.title('Hover Error over Time')
        plt.grid(True)
        plt.show()

        print("\nSimulation completed successfully!")
        print(f"Average iterations per step: {np.mean(iterations):.2f}")
        print(f"Total iterations: {sum(iterations)}")
        if use_rho_adaptation:
            print(f"Final rho: {rho_history[-1]:.2f}")
            print(f"Rho range: [{min(rho_history):.2f}, {max(rho_history):.2f}]")

        # Store final rho for next run
        if use_rho_adaptation and rho_history:
            main.last_rho = rho_history[-1]
            print(f"Saved rho {main.last_rho} for next run")

    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapt', action='store_true', help='Enable rho adaptation')
    parser.add_argument('--recache', action='store_true', help='Enable cache recomputation')
    parser.add_argument('--wind', action='store_true', help='Enable wind disturbance')
    args = parser.parse_args()
    
    main(use_rho_adaptation=args.adapt, use_recaching=args.recache, use_wind=args.wind)