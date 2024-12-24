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
    q = x_curr[3:7]
    phi = QuadrotorDynamics.qtorp(QuadrotorDynamics.L(qg).T @ q)
    delta_x = np.hstack([x_curr[0:3]-rg, phi, x_curr[7:10]-vg, x_curr[10:13]-omgg])
    return delta_x

def tinympc_controller(x_curr, x_nom, u_nom, mpc):
    delta_x = delta_x_quat(x_curr)
    noise = np.zeros(Nx)
    delta_x_noise = (delta_x + noise)  # Remove reshape and tolist

    x_init = np.copy(mpc.x_prev)
    x_init[:,0] = delta_x_noise  # delta_x_noise is already the right shape (12,)
    u_init = np.copy(mpc.u_prev)

    x_out, u_out, status, k = mpc.solve_admm(x_init, u_init, x_nom, u_nom)  
    print(f"Solved with status {status} and k {k}")

    return uhover+u_out[:,0], k

def simulate_with_controller(x0, x_nom, u_nom, mpc, quad, NSIM=100):
    x_all = []
    u_all = []
    x_curr = np.copy(x0)
    iterations = []
    
    for i in range(NSIM):
        u_curr, iters = tinympc_controller(x_curr, x_nom, u_nom, mpc)
        x_curr = quad.dynamics_rk4(x_curr, u_curr)
        
        x_all.append(x_curr)
        u_all.append(u_curr)
        iterations.append(iters)
    
    return np.array(x_all), np.array(u_all), iterations