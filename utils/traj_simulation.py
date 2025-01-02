import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from src.quadrotor import QuadrotorDynamics

# Global reference states
rg = np.array([0.0, 0, 0.0])
qg = np.array([1.0, 0, 0, 0])
vg = np.zeros(3)
omgg = np.zeros(3)
xg = np.hstack([rg, qg, vg, omgg])

# Get hover thrust from QuadrotorDynamics parameters
quad = QuadrotorDynamics()
uhover = quad.hover_thrust
Nx = quad.nx

def delta_x_quat(x_curr, x_ref=None):
    """Compute state error, either from hover or trajectory reference"""
    if x_ref is None:
        # Hover case
        pos_ref = rg
        vel_ref = vg
        omg_ref = omgg
        q_ref = qg
    else:
        # Trajectory following case
        pos_ref = x_ref[0:3]
        vel_ref = x_ref[6:9]  # Note: Make sure these indices match reference generation
        omg_ref = x_ref[9:12]
        q_ref = qg 
    
    # Current state
    q = x_curr[3:7]/np.linalg.norm(x_curr[3:7])  # Normalize quaternion
    
    # Compute attitude error in reduced form (3D)
    phi = QuadrotorDynamics.qtorp(QuadrotorDynamics.L(q_ref).T @ q)
    
    # Compute full error state
    delta_x = np.hstack([
        x_curr[0:3] - pos_ref,    # Position error
        phi,                       # Attitude error (reduced)
        x_curr[7:10] - vel_ref,   # Velocity error
        x_curr[10:13] - omg_ref   # Angular velocity error
    ])
    

    
    # # Add debug prints
    # print("\nError Computation Debug:")
    # print(f"Current position: {x_curr[0:3]}")
    # print(f"Reference position: {x_ref[0:3]}")
    # print(f"Position error: {x_curr[0:3] - x_ref[0:3]}")
    # print(f"Current velocity: {x_curr[7:10]}")
    # print(f"Reference velocity: {x_ref[6:9]}")
    # print(f"Velocity error: {x_curr[7:10] - x_ref[6:9]}")
    
    return delta_x

def tinympc_controller(x_curr, x_nom, u_nom, mpc, t=None, trajectory=None, quad=None):
    """MPC controller for trajectory following"""
    # Generate current reference for error computation
    x_ref = trajectory.generate_reference(t)
    
    # Compute error state
    delta_x = delta_x_quat(x_curr, x_ref)
    
    # Initialize MPC problem with zero reference since we're in error coordinates
    x_init = np.copy(mpc.x_prev)
    x_init[:,0] = delta_x
    u_init = np.copy(mpc.u_prev)
    
    # Solve MPC with zero reference (since we're already in error coordinates)
    x_out, u_out, status, k = mpc.solve_admm(x_init, u_init) 
    
    # Get nominal control
    u_nominal = trajectory.compute_nominal_control(t, quad)
    
    # Debug prints
    print("\nMPC Debug:")
    print(f"Error state: {delta_x}")
    print(f"MPC correction: {u_out[:,0]}")
    print(f"Nominal control: {u_nominal}")
    
    # Combine nominal and correction terms
    u_total = u_nominal + u_out[:,0]
    
    return u_total, k

def simulate_with_controller(x0, x_nom, u_nom, mpc, quad, trajectory, 
                           dt_sim=0.002,   # Explicit simulation timestep
                           dt_mpc=0.02,    # Explicit MPC timestep
                           NSIM=400):      # Number of MPC steps
    """Simulate system with MPC controller
    
    Args:
        x0: Initial state
        x_nom: Initial nominal state trajectory
        u_nom: Initial nominal input trajectory
        mpc: TinyMPC controller instance
        quad: Quadrotor dynamics
        trajectory: Reference trajectory generator
        dt_sim: Simulation timestep (default: 0.002s)
        dt_mpc: MPC update interval (default: 0.02s)
        NSIM: Number of MPC steps (default: 400)
    """
    x_all = []
    u_all = []
    x_curr = np.copy(x0)
    iterations = []
    rho_history = [] if mpc.rho_adapter is not None else None
    current_time = 0.0
    
    # Compute simulation steps per MPC update
    n_sim_steps = int(dt_mpc / dt_sim)

    for i in range(NSIM):
        # Generate goals for entire horizon
        goals = np.zeros((mpc.nx, mpc.N))
        for j in range(mpc.N):
            future_time = current_time + j*dt_mpc
            goals[:,j] = trajectory.generate_reference(future_time)
        
        # Update nominal trajectory using dt_mpc
        for j in range(mpc.N):
            x_nom[:,j] = trajectory.generate_reference(future_time)
            if j < mpc.N-1:
                u_nom[:,j] = trajectory.compute_nominal_control(future_time, quad)
        
        # Run MPC step
        u_curr, iters = tinympc_controller(x_curr, x_nom, u_nom, mpc, 
                                         current_time, trajectory, quad)
        
        # Simulate with finer timestep
        for _ in range(n_sim_steps):
            x_curr = quad.dynamics_rk4(x_curr, u_curr, dt=dt_sim)
            current_time += dt_sim
        
        # Store results
        x_all.append(x_curr)
        u_all.append(u_curr)
        iterations.append(iters)
        
        # Update rho if using adaptation
        if mpc.rho_adapter is not None:
            new_rho = mpc.update_rho()
            rho_history.append(new_rho)
            
        # Shift nominal trajectories with goals
        x_nom, u_nom = mpc.shift_steps(x_nom, u_nom, x_curr, goals=goals)

    # Return results based on whether using rho adaptation
    if mpc.rho_adapter is not None:
        return np.array(x_all), np.array(u_all), iterations, rho_history
    else:
        return np.array(x_all), np.array(u_all), iterations