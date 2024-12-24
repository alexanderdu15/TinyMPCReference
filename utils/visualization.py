# src/utils/visualization.py
import numpy as np
import matplotlib.pyplot as plt

def visualize_trajectory(x_all, u_all, xg, ug):
    """Plot state and input trajectories"""
    x_all = np.array(x_all)
    u_all = np.array(u_all)
    nsteps = len(x_all)
    steps = np.arange(nsteps)

    fig = plt.figure(figsize=(15, 12))
    
    # Position plot
    ax1 = fig.add_subplot(311)
    ax1.plot(steps, x_all[:, 0], label="x", linewidth=2)
    ax1.plot(steps, x_all[:, 1], label="y", linewidth=2)
    ax1.plot(steps, x_all[:, 2], label="z", linewidth=2)
    ax1.plot(steps, [xg[0]]*nsteps, 'r--', label="x_goal")
    ax1.plot(steps, [xg[1]]*nsteps, 'g--', label="y_goal")
    ax1.plot(steps, [xg[2]]*nsteps, 'b--', label="z_goal")
    ax1.set_ylabel('Position [m]')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title("Position Trajectories")

    # Attitude plot
    ax2 = fig.add_subplot(312)
    ax2.plot(steps, x_all[:, 3:7], linewidth=2)
    ax2.plot(steps, [xg[3]]*nsteps, 'r--')
    ax2.set_ylabel('Quaternion')
    ax2.legend(['q0', 'q1', 'q2', 'q3'])
    ax2.grid(True)
    ax2.set_title("Attitude Trajectories")

    # Control inputs plot
    ax3 = fig.add_subplot(313)
    ax3.plot(steps, u_all, linewidth=2)
    ax3.plot(steps, [ug[0]]*nsteps, 'k--', label="hover_thrust")
    ax3.set_xlabel('Time steps')
    ax3.set_ylabel('Motor commands')
    ax3.legend(['u1', 'u2', 'u3', 'u4'])
    ax3.grid(True)
    ax3.set_title("Control Inputs")

    plt.tight_layout()
    plt.show()

def plot_iterations(iterations):
    """Plot ADMM iterations"""
    plt.figure(figsize=(10, 4))
    plt.plot(iterations, label='Iterations')
    plt.ylabel('Iterations')
    plt.title('ADMM Iterations per Time Step')
    plt.grid(True)
    plt.legend()
    plt.show()
    print(f"Total iterations: {sum(iterations)}")

def plot_rho_history(rho_history):
    """Plot rho adaptation history (only for adaptive MPC)"""
    plt.figure(figsize=(10, 4))
    plt.plot(rho_history, label='Rho')
    plt.xlabel('Time Step')
    plt.ylabel('Rho Value')
    plt.title('Rho Adaptation History')
    plt.grid(True)
    plt.legend()
    plt.show()