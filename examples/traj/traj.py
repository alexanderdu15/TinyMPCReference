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
import argparse

def compute_tracking_error(x_all, trajectory, dt):
    """Compute L2 tracking error over the trajectory"""
    errors = []
    times = np.arange(len(x_all)) * dt
    
    for t, x in zip(times, x_all):
        x_ref = trajectory.generate_reference(t)
        pos_error = np.linalg.norm(x[0:3] - x_ref[0:3])
        errors.append(pos_error)
    
    return np.mean(errors), np.max(errors), errors

def main(use_rho_adaptation=False, use_recaching=False, use_wind=False, traj_type='full'):
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
    w = 2*np.pi/3.0
    trajectory = Figure8Reference(A=amplitude, w=w, segment_type=traj_type)
    
    # Get the initial reference point and set initial state
    x_ref_0 = trajectory.generate_reference(0.0)
    x0[0:3] = x_ref_0[0:3]     # Position
    x0[3:7] = qg               # Level quaternion
    x0[7:10] = x_ref_0[6:9]    # All velocities (not just x and z)
    x0[10:13] = np.zeros(3)    # Zero angular velocity

    # Debug print to verify initial state
    print("\nInitial State:")
    print(f"Position: {x0[0:3]}")
    print(f"Quaternion: {x0[3:7]}")
    print(f"Velocity: {x0[7:10]}")
    print(f"Angular velocity: {x0[10:13]}")

    # Cost matrices
    max_dev_x = np.array([
        0.01, 0.01, 0.01,    # position (tighter)
        0.5, 0.5, 0.05,      # attitude 
        0.5, 0.5, 0.5,       # velocity
        0.7, 0.7, 0.5        # angular velocity
    ])
    max_dev_u = np.array([0.1, 0.1, 0.1, 0.1])

    Q = np.diag(1.0 / (max_dev_x**2))  # Much higher position weights
    R = np.diag(1.0 / (max_dev_u**2))

    # Setup PC
    N = 15
    initial_rho = 1.0

    rho_adapter = None
    if use_rho_adaptation:
        rho_adapter = RhoAdapter(rho_base=initial_rho, rho_min=1.0, rho_max=200.0)
       

    # Initialize MPC
    mpc = TinyMPC(
        A=A,
        B=B,
        Q=Q,
        R=R,
        Nsteps=N,
        rho=initial_rho,
        rho_adapter= rho_adapter,
        recache = use_recaching
    )

   
    if use_rho_adaptation:
        mpc.rho_adapter.initialize_derivatives(mpc.cache)
       

    # Set bounds relative to hover thrust
    u_max = np.array([0.3] * quad.nu)  # Allow 30% above hover
    u_min = np.array([-0.3] * quad.nu)  # Allow 30% below hover
    x_max = [2.0] * 3 + [1.0] * 3 + [2.0] * 3 + [2.0] * 3  # [pos, att, vel, omega]
    x_min = [-x for x in x_max]
    mpc.set_bounds(u_max, u_min, x_max, x_min)

    # Create trajectory reference (keep your original parameters)
    x_nom = np.zeros((quad.nx, mpc.N))
    u_nom = np.zeros((quad.nu, mpc.N-1))

    u_nom[:] = ug.reshape(-1,1)  # hover thrust

    # Initialize nominal states from trajectory
    x_nom = np.zeros((quad.nx, mpc.N))
    for i in range(mpc.N):
        x_nom[:,i] = trajectory.generate_reference(i*quad.dt)

    # Debug prints
    print("\nNominal control check:")
    print(f"Hover thrust: {quad.hover_thrust}")
    print(f"Initial u_nom: {u_nom[:,0]}")
    print(f"Mid-horizon u_nom: {u_nom[:,mpc.N//2]}")

    # Run simulation
    try:
        print(f"Starting trajectory tracking simulation with:")
        print(f"- Rho adaptation: {'enabled' if use_rho_adaptation else 'disabled'}")
        print(f"- Cache recomputation: {'enabled' if use_recaching else 'disabled'}")

        simulation_result = simulate_with_controller(
            x0=x0,
            x_nom=x_nom,
            u_nom=u_nom,
            mpc=mpc,
            quad=quad,
            trajectory=trajectory,
            dt_sim=0.01,
            #dt_mpc=quad.dt,
            dt_mpc=0.02,
            NSIM=300,
            use_wind=use_wind
        )

        # Unpack results based on whether we're using rho adaptation
        if use_rho_adaptation:
            x_all, u_all, iterations, rho_history, metrics = simulation_result
        else:
            x_all, u_all, iterations, metrics = simulation_result
            rho_history = None

        # Now you can access the metrics separately
        trajectory_costs = metrics['trajectory_costs']
        control_efforts = metrics['control_efforts']

        

        # Compute and display tracking error
        avg_error, max_error, errors = compute_tracking_error(x_all, trajectory, quad.dt)
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
        suffix += f'_{traj_type}'  # Add trajectory type to suffix

        for dir_name in ['iterations', 'rho_history', 'trajectory_costs', 'control_efforts']:
            (data_dir / dir_name).mkdir(exist_ok=True)

        np.savetxt(data_dir / 'iterations' / f"traj{suffix}.txt", iterations)
        
        if use_rho_adaptation:
            np.savetxt(data_dir / 'rho_history' / f"traj{suffix}.txt", rho_history)

        # Save new metrics
        np.savetxt(data_dir / 'trajectory_costs' / f"traj{suffix}.txt", metrics['trajectory_costs'])
        np.savetxt(data_dir / 'control_efforts' / f"traj{suffix}.txt", metrics['control_efforts'])

        # Visualize results
        visualize_trajectory(x_all, u_all, trajectory=trajectory, dt=quad.dt)
        plot_iterations(iterations)
        if use_rho_adaptation:
            plot_rho_history(rho_history)

        # Plot tracking error
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(len(errors))*quad.dt, errors)
        plt.xlabel('Time [s]')
        plt.ylabel('L2 Position Error [m]')
        plt.title('Tracking Error over Time')
        plt.grid(True)

        # Plot new metrics
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(len(metrics['trajectory_costs']))*quad.dt, 
                 metrics['trajectory_costs'], label='Trajectory Cost')
        plt.xlabel('Time [s]')
        plt.ylabel('Cost')
        plt.title('Trajectory Cost over Time')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(np.arange(len(metrics['control_efforts']))*quad.dt, 
                 metrics['control_efforts'], label='Control Effort')
        plt.xlabel('Time [s]')
        plt.ylabel('Total Torque')
        plt.title('Control Effort over Time')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print("\nSimulation completed successfully!")
        print(f"Average iterations per step: {np.mean(iterations):.2f}")
        print(f"Total iterations: {sum(iterations)}")
        print(f"Average trajectory cost: {np.mean(metrics['trajectory_costs']):.4f}")
        print(f"Average control effort: {np.mean(metrics['control_efforts']):.4f}")

    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapt', action='store_true', 
                        help='Enable rho adaptation')
    parser.add_argument('--recache', action='store_true', 
                        help='Enable cache recomputation')
    parser.add_argument('--wind', action='store_true', 
                        help='Enable wind disturbance')
    
    # Mutually exclusive group for trajectory type
    traj_group = parser.add_mutually_exclusive_group()
    traj_group.add_argument('--straight', action='store_true', 
                           help='Run straight segments only')
    traj_group.add_argument('--curve', action='store_true', 
                           help='Run curved segments only')
    
    args = parser.parse_args()
    
    # Determine trajectory type
    if args.straight:
        traj_type = 'straight'
    elif args.curve:
        traj_type = 'curve'
    else:
        traj_type = 'full'  # default
    
    main(use_rho_adaptation=args.adapt, 
         use_recaching=args.recache,
         use_wind=args.wind,
         traj_type=traj_type)