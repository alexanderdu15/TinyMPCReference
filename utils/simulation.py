# utils/simulation.py
import numpy as np

def tinympc_controller(x_curr, x_nom, u_nom, mpc):
    """MPC controller wrapper"""
    x_sol, u_sol, status, iters = mpc.solve_admm(x_curr, x_nom, u_nom)
    return u_sol[:, 0], iters

def simulate_with_controller(x0, x_nom, u_nom, mpc, quad, NSIM=100):
    """Simulate system with given controller"""
    x_all = []
    u_all = []
    x_curr = np.copy(x0)
    iterations = []
    
    for i in range(NSIM):
        u_curr, iters = tinympc_controller(x_curr, x_nom, u_nom, mpc)
        u_curr_clipped = np.clip(u_curr, 0, 1)
        x_curr = quad.dynamics_rk4(x_curr, u_curr_clipped)
        
        x_all.append(x_curr)
        u_all.append(u_curr)
        iterations.append(iters)
    
    return np.array(x_all), np.array(u_all), iterations