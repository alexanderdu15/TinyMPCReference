import numpy as np
import matplotlib.pyplot as plt

def visualize_trajectory(x_all, u_all, xg=None, ug=None, trajectory=None, dt=0.02):
    """
    Plot state and input trajectories 
    """
    x_all = np.array(x_all)
    u_all = np.array(u_all)
    nsteps = len(x_all)
    t = np.arange(nsteps) * dt

    # Get reference trajectory once if it exists
    if trajectory is not None:
        x_ref = trajectory.get_trajectory_points(t)

    fig = plt.figure(figsize=(15, 12))
    
    # 2D trajectory plot
    ax0 = fig.add_subplot(221)
    ax0.plot(x_all[:, 0], x_all[:, 2], 'b-', label='Actual', linewidth=2)
    
    if trajectory is not None:
        ax0.plot(x_ref[:, 0], x_ref[:, 2], 'r--', label='Reference', linewidth=2)
    
    ax0.set_xlabel('X [m]')
    ax0.set_ylabel('Z [m]')
    ax0.legend()
    ax0.grid(True)
    ax0.set_title("2D Trajectory")
    
    # Position plot
    ax1 = fig.add_subplot(222)
    ax1.plot(t, x_all[:, 0], 'b-', label="x", linewidth=2)
    ax1.plot(t, x_all[:, 1], 'g-', label="y", linewidth=2)
    ax1.plot(t, x_all[:, 2], 'r-', label="z", linewidth=2)
    
    if trajectory is not None:
        # Use same reference trajectory data
        ax1.plot(t, x_ref[:, 0], 'b--', label="x_ref")
        ax1.plot(t, x_ref[:, 2], 'r--', label="z_ref")
    elif xg is not None:
        ax1.plot(t, [xg[0]]*nsteps, 'b--', label="x_goal")
        ax1.plot(t, [xg[1]]*nsteps, 'g--', label="y_goal")
        ax1.plot(t, [xg[2]]*nsteps, 'r--', label="z_goal")
    
    ax1.set_ylabel('Position [m]')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title("Position Trajectories")

    # Attitude plot
    ax2 = fig.add_subplot(223)
    ax2.plot(t, x_all[:, 3:7], linewidth=2)
    if xg is not None:
        ax2.plot(t, [xg[3]]*nsteps, 'r--')
    ax2.set_ylabel('Quaternion')
    ax2.legend(['q0', 'q1', 'q2', 'q3'])
    ax2.grid(True)
    ax2.set_title("Attitude Trajectories")

    # Control inputs plot
    ax3 = fig.add_subplot(224)
    ax3.plot(t, u_all, linewidth=2)
    if ug is not None:
        ax3.plot(t, [ug[0]]*nsteps, 'k--', label="hover_thrust")
    ax3.set_xlabel('Time [s]')
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
    print(f"Average iterations: {np.mean(iterations):.2f}")

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
    print(f"Final rho: {rho_history[-1]:.2f}")
    print(f"Rho range: [{min(rho_history):.2f}, {max(rho_history):.2f}]")