# utils/simulation.py
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

def delta_x_quat(x_curr):
    """Compute state error for hover"""
    q = x_curr[3:7]/np.linalg.norm(x_curr[3:7])  # Normalize quaternion
    phi = QuadrotorDynamics.qtorp(QuadrotorDynamics.L(qg).T @ q)
    delta_x = np.hstack([
        x_curr[0:3]-rg,     # Position error
        phi,                # Attitude error
        x_curr[7:10]-vg,    # Velocity error
        x_curr[10:13]-omgg  # Angular velocity error
    ])
    return delta_x

def tinympc_controller(x_curr, x_nom, u_nom, mpc):
    """MPC controller for hover"""
    # Compute error state
    delta_x = delta_x_quat(x_curr)
    
    # Initialize MPC problem
    x_init = np.copy(mpc.x_prev)
    x_init[:,0] = delta_x
    u_init = np.copy(mpc.u_prev)

    # Solve MPC with zero reference (since we're in error coordinates)
    x_out, u_out, status, k = mpc.solve_admm(x_init, u_init)
    print(f"Solved with status {status} and k {k}")

    # Return control with hover thrust
    return uhover + u_out[:,0], k

def simulate_with_controller(x0, x_nom, u_nom, mpc, quad, NSIM=100):
    """Simulate system with MPC controller"""
    x_all = []
    u_all = []
    x_curr = np.copy(x0)
    iterations = []
    rho_history = [] if mpc.rho_adapter is not None else None
    
    for i in range(NSIM):
        # Run MPC step
        u_curr, iters = tinympc_controller(x_curr, x_nom, u_nom, mpc)
        
        # Simulate system
        x_curr = quad.dynamics_rk4(x_curr, u_curr)
        
        # Store results
        x_all.append(x_curr)
        u_all.append(u_curr)
        iterations.append(iters)
        
        # Update rho if adaptation is enabled
        if mpc.rho_adapter is not None:
            new_rho = mpc.update_rho()
            rho_history.append(new_rho)

    # Return results based on whether rho adaptation is enabled
    if mpc.rho_adapter is not None:
        return np.array(x_all), np.array(u_all), iterations, rho_history
    else:
        return np.array(x_all), np.array(u_all), iterations