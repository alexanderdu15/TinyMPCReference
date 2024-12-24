# examples/hover/hover.py
import sys
from pathlib import Path
# Add TinyMPCReference root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.quadrotor import *
from src.tinympc import TinyMPC

def main():
    # Initialize system
    rg = np.array([0.0, 0, 0.0])
    qg = np.array([1.0, 0, 0, 0])
    vg = np.zeros(3)
    omgg = np.zeros(3)
    xg = np.hstack([rg, qg, vg, omgg])
    uhover = (mass*g/kt/4)*np.ones(4)

    # Get linearized system
    A_jac = jacobian(quad_dynamics_rk4, 0)
    B_jac = jacobian(quad_dynamics_rk4, 1)
    Anp1 = A_jac(xg, uhover)
    Bnp1 = B_jac(xg, uhover)
    Anp = E(qg).T @ Anp1 @ E(qg)
    Bnp = E(qg).T @ Bnp1

    # Setup MPC
    N = 10
    input_data = {
        'rho': 85.0,
        'A': Anp,
        'B': Bnp,
        'Q': P_lqr,
        'R': R
    }

    tinympc = TinyMPC(input_data, N)
    tinympc.set_bounds(u_max, u_min, x_max, x_min)

    # Run simulation
    x_all, u_all, iterations = simulate_with_controller(x0, x_nom, u_nom, tinympc_controller)

    # Visualize results
    visualize_trajectory(x_all, u_all)
    plot_iterations(iterations)

if __name__ == "__main__":
    main()