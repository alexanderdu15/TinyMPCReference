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
    q = x_curr[3:7]
    
    if x_ref is None:
        # Hover case
        pos_ref = rg
        vel_ref = vg
        omg_ref = omgg
        q_ref = qg
    else:
        # Trajectory following case
        pos_ref = x_ref[0:3]
        vel_ref = x_ref[6:9]  # Changed from 7:10 to 6:9
        omg_ref = x_ref[9:12]  # Changed from 10:13 to 9:12
        q_ref = np.array([1.0, 0.0, 0.0, 0.0])  # Assume upright orientation
    
    phi = QuadrotorDynamics.qtorp(QuadrotorDynamics.L(q_ref).T @ q)
    delta_x = np.hstack([
        x_curr[0:3] - pos_ref,
        phi,
        x_curr[7:10] - vel_ref,
        x_curr[10:13] - omg_ref
    ])
    return delta_x

def tinympc_controller(x_curr, x_nom, u_nom, mpc, t=None, trajectory=None):
    """MPC controller for trajectory following"""
    # Generate current reference for error computation
    x_ref = trajectory.generate_reference(t)
    
    # Compute error
    delta_x = delta_x_quat(x_curr, x_ref)
    delta_x_noise = delta_x

    # Initialize MPC problem
    x_init = np.copy(mpc.x_prev)
    x_init[:,0] = delta_x_noise
    u_init = np.copy(mpc.u_prev)

    # Solve MPC using provided nominal trajectory
    x_out, u_out, status, k = mpc.solve_admm(x_init, u_init, x_nom, u_nom)
    print(f"Solved with status {status} and k {k}")

    return uhover+u_out[:,0], k

def simulate_with_controller(x0, x_nom, u_nom, mpc, quad, trajectory, NSIM=400):
    """Simulation loop for trajectory tracking"""
    x_all = []
    u_all = []
    x_curr = np.copy(x0)
    iterations = []
    
    for i in range(NSIM):
        t = i * quad.dt
        
        # Run MPC step with provided nominal trajectory
        u_curr, iters = tinympc_controller(x_curr, x_nom, u_nom, mpc, t, trajectory)
        
        # Simulate system
        x_curr = quad.dynamics_rk4(x_curr, u_curr)
        
        x_all.append(x_curr)
        u_all.append(u_curr)
        iterations.append(iters)
    
    return np.array(x_all), np.array(u_all), iterations