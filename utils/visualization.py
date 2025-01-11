import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

def plot_costs_comparison(adaptive_costs, fixed_costs):
    """Plot comparison of costs between adaptive and fixed rho"""
    plt.figure(figsize=(15, 5))
    
    # State Costs
    plt.subplot(131)
    plt.plot(adaptive_costs[:, 0], label='Adaptive', alpha=0.8)
    plt.plot(fixed_costs[:, 0], label='Fixed', alpha=0.8)
    plt.xlabel('MPC Step')
    plt.ylabel('State Cost')
    plt.title('State Costs Comparison')
    plt.legend()
    plt.grid(True)
    
    # Input Costs
    plt.subplot(132)
    plt.plot(adaptive_costs[:, 1], label='Adaptive', alpha=0.8)
    plt.plot(fixed_costs[:, 1], label='Fixed', alpha=0.8)
    plt.xlabel('MPC Step')
    plt.ylabel('Input Cost')
    plt.title('Input Costs Comparison')
    plt.legend()
    plt.grid(True)
    
    # Statistics
    plt.subplot(133)
    stats_text = (
        f'Average State Cost:\n'
        f'Adaptive: {np.mean(adaptive_costs[:, 0]):.3f}\n'
        f'Fixed: {np.mean(fixed_costs[:, 0]):.3f}\n\n'
        f'Average Input Cost:\n'
        f'Adaptive: {np.mean(adaptive_costs[:, 1]):.3f}\n'
        f'Fixed: {np.mean(fixed_costs[:, 1]):.3f}'
    )
    plt.text(0.1, 0.1, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.axis('off')
    plt.title('Cost Statistics')
    
    plt.tight_layout()
    plt.show()

def plot_violations_comparison(adaptive_violations, fixed_violations):
    """Plot comparison of constraint violations between adaptive and fixed rho"""
    plt.figure(figsize=(15, 5))
    
    # Input Violations
    plt.subplot(131)
    plt.plot(adaptive_violations[:, 0], label='Adaptive', alpha=0.8)
    plt.plot(fixed_violations[:, 0], label='Fixed', alpha=0.8)
    plt.xlabel('MPC Step')
    plt.ylabel('Input Constraint Violation')
    plt.title('Input Violations Comparison')
    plt.legend()
    plt.grid(True)
    
    # State Violations
    plt.subplot(132)
    plt.plot(adaptive_violations[:, 1], label='Adaptive', alpha=0.8)
    plt.plot(fixed_violations[:, 1], label='Fixed', alpha=0.8)
    plt.xlabel('MPC Step')
    plt.ylabel('State Constraint Violation')
    plt.title('State Violations Comparison')
    plt.legend()
    plt.grid(True)
    
    # Statistics
    plt.subplot(133)
    stats_text = (
        f'Average Input Violation:\n'
        f'Adaptive: {np.mean(adaptive_violations[:, 0]):.3f}\n'
        f'Fixed: {np.mean(fixed_violations[:, 0]):.3f}\n\n'
        f'Average State Violation:\n'
        f'Adaptive: {np.mean(adaptive_violations[:, 1]):.3f}\n'
        f'Fixed: {np.mean(fixed_violations[:, 1]):.3f}'
    )
    plt.text(0.1, 0.1, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.axis('off')
    plt.title('Violation Statistics')
    
    plt.tight_layout()
    plt.show()

def save_metrics(metrics, suffix, data_dir='data'):
    """Save metrics to files"""
    data_dir = Path(data_dir)
    for subdir in ['costs', 'violations', 'iterations']:
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    np.savetxt(data_dir / 'costs' / f'costs_{suffix}.txt', metrics['solve_costs'])
    np.savetxt(data_dir / 'violations' / f'violations_{suffix}.txt', metrics['violations'])
    np.savetxt(data_dir / 'iterations' / f'iterations_{suffix}.txt', metrics['iterations'])

def plot_all_metrics(metrics, iterations, errors, rho_history=None, use_rho_adaptation=False, dt=0.02):
    """Plot all metrics in one figure except state/cost plots"""
    plt.figure(figsize=(20, 10))
    
    # Plot tracking error
    plt.subplot(231)
    plt.plot(np.arange(len(errors))*dt, errors)
    plt.xlabel('Time [s]')
    plt.ylabel('L2 Position Error [m]')
    plt.title('Tracking Error over Time')
    plt.grid(True)
    
    # Plot violations
    plt.subplot(232)
    violations = np.array(metrics['violations'])
    plt.plot(violations[:, 0], label='Input Violation')
    plt.plot(violations[:, 1], label='State Violation')
    plt.xlabel('MPC Step')
    plt.ylabel('Constraint Violation')
    plt.title('Constraint Violations')
    plt.legend()
    plt.grid(True)
    
    # Plot iterations
    plt.subplot(233)
    plt.plot(iterations, label='Iterations')
    plt.xlabel('MPC Step')
    plt.ylabel('Iterations')
    plt.title('ADMM Iterations')
    plt.grid(True)
    
    # Plot trajectory costs
    plt.subplot(234)
    plt.plot(metrics['trajectory_costs'], label='Trajectory Cost')
    plt.xlabel('MPC Step')
    plt.ylabel('Cost')
    plt.title('Trajectory Costs')
    plt.grid(True)
    
    # Plot control efforts
    plt.subplot(235)
    plt.plot(metrics['control_efforts'], label='Control Effort')
    plt.xlabel('MPC Step')
    plt.ylabel('Effort')
    plt.title('Control Efforts')
    plt.grid(True)
    
    # Plot rho history if adaptive
    plt.subplot(236)
    if use_rho_adaptation and rho_history is not None:
        plt.plot(rho_history, label='Rho')
        plt.xlabel('Time Step')
        plt.ylabel('Rho Value')
        plt.title('Rho Adaptation History')
    else:
        plt.text(0.5, 0.5, 'Statistics:\n' + 
                f"Avg iterations: {np.mean(iterations):.2f}\n" +
                f"Total iterations: {sum(iterations)}\n" +
                f"Avg traj cost: {np.mean(metrics['trajectory_costs']):.4f}\n" +
                f"Avg control effort: {np.mean(metrics['control_efforts']):.4f}",
                horizontalalignment='center',
                verticalalignment='center',
                transform=plt.gca().transAxes)
        plt.axis('off')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_state_and_costs(metrics, use_rho_adaptation=False):
    """Plot state and cost metrics separately"""
    plt.figure(figsize=(15, 5))
    
    # Plot costs
    plt.subplot(121)
    costs = np.array(metrics['solve_costs'])
    plt.plot(costs[:, 0], label='State Cost')
    plt.plot(costs[:, 1], label='Input Cost')
    plt.plot(costs[:, 2], label='Total Cost')
    plt.xlabel('MPC Step')
    plt.ylabel('Cost')
    plt.title(f'{"Adaptive" if use_rho_adaptation else "Fixed"} Rho Costs')
    plt.legend()
    plt.grid(True)
    
    # Plot state violations
    plt.subplot(122)
    violations = np.array(metrics['violations'])
    plt.plot(violations[:, 1], label='State Violation')
    plt.xlabel('MPC Step')
    plt.ylabel('State Constraint Violation')
    plt.title('State Violations')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_comparisons(data_dir='../data', traj_type='full', compare_type='normal'):
    """
    Load saved data and create comparison plots
    compare_type: 'normal' (adaptive vs fixed) or 'wind' (adaptive+wind vs fixed+wind)
    """
    try:
        data_dir = Path(data_dir)
        print("\nLoading metrics for comparison...")
        
        if compare_type == 'wind':
            # Load adaptive with wind data
            adaptive_costs = np.loadtxt(data_dir / 'costs' / f'costs_adaptive_wind_{traj_type}.txt')
            adaptive_violations = np.loadtxt(data_dir / 'violations' / f'violations_adaptive_wind_{traj_type}.txt')
            adaptive_traj_costs = np.loadtxt(data_dir / 'trajectory_costs' / f'traj_adaptive_wind_{traj_type}.txt')
            adaptive_control = np.loadtxt(data_dir / 'control_efforts' / f'traj_adaptive_wind_{traj_type}.txt')
            
            # Load fixed with wind data
            fixed_costs = np.loadtxt(data_dir / 'costs' / f'costs_normal_wind_{traj_type}.txt')
            fixed_violations = np.loadtxt(data_dir / 'violations' / f'violations_normal_wind_{traj_type}.txt')
            fixed_traj_costs = np.loadtxt(data_dir / 'trajectory_costs' / f'traj_normal_wind_{traj_type}.txt')
            fixed_control = np.loadtxt(data_dir / 'control_efforts' / f'traj_normal_wind_{traj_type}.txt')
            
            title_suffix = 'with Wind'
        else:  # normal comparison (adaptive vs fixed, no wind)
            # Load adaptive data (no wind)
            adaptive_costs = np.loadtxt(data_dir / 'costs' / f'costs_adaptive_{traj_type}.txt')
            adaptive_violations = np.loadtxt(data_dir / 'violations' / f'violations_adaptive_{traj_type}.txt')
            adaptive_traj_costs = np.loadtxt(data_dir / 'trajectory_costs' / f'traj_adaptive_{traj_type}.txt')
            adaptive_control = np.loadtxt(data_dir / 'control_efforts' / f'traj_adaptive_{traj_type}.txt')
            
            # Load fixed data (no wind)
            fixed_costs = np.loadtxt(data_dir / 'costs' / f'costs_normal_{traj_type}.txt')
            fixed_violations = np.loadtxt(data_dir / 'violations' / f'violations_normal_{traj_type}.txt')
            fixed_traj_costs = np.loadtxt(data_dir / 'trajectory_costs' / f'traj_normal_{traj_type}.txt')
            fixed_control = np.loadtxt(data_dir / 'control_efforts' / f'traj_normal_{traj_type}.txt')
            
            title_suffix = 'no Wind'
        
        # Create figure with extra space on right for stats
        plt.figure(figsize=(22, 10))  # Made figure wider to accommodate stats
        
        # State Costs
        plt.subplot(231)
        plt.plot(fixed_costs[:, 0], 'b-o', label='Fixed', alpha=0.5, linewidth=2, markersize=4, markevery=20)
        plt.plot(adaptive_costs[:, 0], 'r-^', label='Adaptive', alpha=0.8, linewidth=2.5, markersize=6, markevery=20)
        plt.xlabel('MPC Step')
        plt.ylabel('State Cost')
        plt.title(f'State Costs Comparison ({title_suffix})')
        plt.legend()
        plt.grid(True)
        
        # Input Costs
        plt.subplot(232)
        plt.plot(fixed_costs[:, 1], 'b-s', label='Fixed', alpha=0.5, linewidth=2, markersize=4, markevery=20)
        plt.plot(adaptive_costs[:, 1], 'r-d', label='Adaptive', alpha=0.8, linewidth=2.5, markersize=6, markevery=20)
        plt.xlabel('MPC Step')
        plt.ylabel('Input Cost')
        plt.title('Input Costs Comparison')
        plt.legend()
        plt.grid(True)
        
        # Input Violations
        plt.subplot(233)
        plt.plot(fixed_violations[:, 0], 'b-*', label='Fixed', alpha=0.5, linewidth=2, markersize=4, markevery=20)
        plt.plot(adaptive_violations[:, 0], 'r-p', label='Adaptive', alpha=0.8, linewidth=2.5, markersize=6, markevery=20)
        plt.xlabel('MPC Step')
        plt.ylabel('Input Constraint Violation')
        plt.title('Input Violations')
        plt.legend()
        plt.grid(True)

        # State Violations (with adjusted scale)
        plt.subplot(234)
        plt.plot(fixed_violations[:, 1], 'b-*', label='Fixed', alpha=0.5, linewidth=2, markersize=4, markevery=20)
        plt.plot(adaptive_violations[:, 1], 'r-p', label='Adaptive', alpha=0.8, linewidth=2.5, markersize=6, markevery=20)
        plt.xlabel('MPC Step')
        plt.ylabel('State Constraint Violation')
        plt.title('State Violations (×10⁻³)')
        plt.legend()
        plt.grid(True)
        
        # Control Efforts
        plt.subplot(235)
        plt.plot(fixed_control, 'b-x', label='Fixed', alpha=0.5, linewidth=2, markersize=4, markevery=20)
        plt.plot(adaptive_control, 'r-+', label='Adaptive', alpha=0.8, linewidth=2.5, markersize=6, markevery=20)
        plt.xlabel('MPC Step')
        plt.ylabel('Control Effort')
        plt.title('Control Efforts')
        plt.legend()
        plt.grid(True)
        
        
        
        # Trajectory Costs
        plt.subplot(236)
        plt.plot(fixed_traj_costs, 'b-v', label='Fixed', alpha=0.5, linewidth=2, markersize=4, markevery=20)
        plt.plot(adaptive_traj_costs, 'r-^', label='Adaptive', alpha=0.8, linewidth=2.5, markersize=6, markevery=20)
        plt.xlabel('MPC Step')
        plt.ylabel('Trajectory Cost')
        plt.title('Trajectory Costs')
        plt.legend()
        plt.grid(True)
        
        # Add statistics box to the right of plots
        stats_text = (
            f'Average Metrics:\n\n'
            f'State Cost:\n'
            f'  Fixed: {np.mean(fixed_costs[:, 0]):.3f}\n'
            f'  Adaptive: {np.mean(adaptive_costs[:, 0]):.3f}\n\n'
            f'Input Cost:\n'
            f'  Fixed: {np.mean(fixed_costs[:, 1]):.3f}\n'
            f'  Adaptive: {np.mean(adaptive_costs[:, 1]):.3f}\n\n'
            f'Total Cost:\n'
            f'  Fixed: {np.mean(fixed_costs[:, 0] + fixed_costs[:, 1]):.3f}\n'
            f'  Adaptive: {np.mean(adaptive_costs[:, 0] + adaptive_costs[:, 1]):.3f}\n\n'
            f'Input Violation:\n'
            f'  Fixed: {np.mean(fixed_violations[:, 0]):.3f}\n'
            f'  Adaptive: {np.mean(adaptive_violations[:, 0]):.3f}\n\n'
            f'State Violation:\n'
            f'  Fixed: {np.mean(fixed_violations[:, 1]):.3f}\n'
            f'  Adaptive: {np.mean(adaptive_violations[:, 1]):.3f}\n\n'
            f'Total Violation:\n'
            f'  Fixed: {np.mean(fixed_violations[:, 0] + fixed_violations[:, 1]):.3f}\n'
            f'  Adaptive: {np.mean(adaptive_violations[:, 0] + adaptive_violations[:, 1]):.3f}\n\n'
            f'Trajectory Cost:\n'
            f'  Fixed: {np.mean(fixed_traj_costs):.3f}\n'
            f'  Adaptive: {np.mean(adaptive_traj_costs):.3f}'
        )
        # Place stats box to the right of plots
        plt.figtext(0.92, 0.5, stats_text, 
                   fontsize=10,
                   bbox=dict(facecolor='white', edgecolor='black', alpha=0.8),
                   verticalalignment='center')
        
        plt.tight_layout()
        # Adjust layout to make room for stats box
        plt.subplots_adjust(right=0.88)
        plt.show()
        
    except FileNotFoundError as e:
        print("\nError: Make sure you've run both cases before comparison")
        print(f"Missing file: {e}")