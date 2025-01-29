import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_trajectory(x_all, u_all, trajectory=None, dt=0.02):
    """Visualize trajectory with early stopping and clearer plotting"""
    x_all = np.array(x_all)
    u_all = np.array(u_all)
    t = np.arange(len(x_all)) * dt
    t_full = np.arange(200) * dt  # Full trajectory time

    # Find divergence point (position error > 5m or NaN)
    divergence_idx = None
    if trajectory is not None:
        for i, x in enumerate(x_all):
            x_ref = trajectory.generate_reference(t[i])[0:3]
            pos_error = np.linalg.norm(x[0:3] - x_ref)
            if pos_error > 5.0 or np.any(np.isnan(x)):
                divergence_idx = i
                print(f"\nController diverged at t = {t[i]:.2f} seconds")
                print(f"Position error: {pos_error:.2f} meters")
                break
    
    # Trim data if divergence detected
    if divergence_idx is not None:
        x_all = x_all[:divergence_idx]
        u_all = u_all[:divergence_idx]
        t = t[:divergence_idx]

    # Create time series plots
    fig1, axes = plt.subplots(3, 1, figsize=(12, 10), dpi=100)
    
    # Position plot with full reference trajectory
    axes[0].plot(t, x_all[:, 0], 'r-', label='x', linewidth=2)
    axes[0].plot(t, x_all[:, 1], 'g-', label='y', linewidth=2)
    axes[0].plot(t, x_all[:, 2], 'b-', label='z', linewidth=2)
    if trajectory is not None:
        # Plot full reference trajectory
        x_ref_full = np.array([trajectory.generate_reference(ti)[0:3] for ti in t_full])
        axes[0].plot(t_full, x_ref_full[:, 0], 'r--', label='x ref', linewidth=2)
        axes[0].plot(t_full, x_ref_full[:, 1], 'g--', label='y ref', linewidth=2)
        axes[0].plot(t_full, x_ref_full[:, 2], 'b--', label='z ref', linewidth=2)
    axes[0].set_ylabel('Position (m)', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True)
    axes[0].tick_params(labelsize=10)

    # Attitude plot
    axes[1].plot(t, x_all[:, 3], 'r-', label='qw', linewidth=2)
    axes[1].plot(t, x_all[:, 4], 'g-', label='qx', linewidth=2)
    axes[1].plot(t, x_all[:, 5], 'b-', label='qy', linewidth=2)
    axes[1].plot(t, x_all[:, 6], 'k-', label='qz', linewidth=2)
    axes[1].set_ylabel('Attitude (quat)', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True)
    axes[1].tick_params(labelsize=10)

    # Control plot
    t_control = t[:len(u_all)]  # Adjust time vector for control inputs
    axes[2].plot(t_control, u_all[:, 0], 'r-', label='u1', linewidth=2)
    axes[2].plot(t_control, u_all[:, 1], 'g-', label='u2', linewidth=2)
    axes[2].plot(t_control, u_all[:, 2], 'b-', label='u3', linewidth=2)
    axes[2].plot(t_control, u_all[:, 3], 'k-', label='u4', linewidth=2)
    axes[2].set_ylabel('Control', fontsize=12)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(True)
    axes[2].tick_params(labelsize=10)

    plt.tight_layout()

    # Create 2D trajectory plot (X-Z plane)
    fig2, ax = plt.subplots(figsize=(10, 8), dpi=200)
    
    # Plot actual trajectory
    ax.plot(x_all[:, 0], x_all[:, 2], 'b-', label='Actual', linewidth=2)
    
    # Plot reference trajectory if available
    if trajectory is not None:
        t_full = np.linspace(0, 4, 1000)  # Smooth reference trajectory
        x_ref_full = np.array([trajectory.generate_reference(ti)[0:3] for ti in t_full])
        ax.plot(x_ref_full[:, 0], x_ref_full[:, 2], 'r--', linewidth=2)
    
    ax.set_xticklabels([])  # Remove x ticks
    ax.set_yticklabels([])  # Remove y ticks
    ax.set_aspect('equal')  # Keep square aspect ratio
    
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
    plt.title('State Costs Comparison')
    plt.legend()
    plt.grid(True)
    
    # Input Costs
    plt.subplot(132)
    plt.plot(adaptive_costs[:, 1], label='Adaptive', alpha=0.8)
    plt.plot(fixed_costs[:, 1], label='Fixed', alpha=0.8)
    plt.xlabel('MPC Step')
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

def plot_state_and_costs(suffix, use_rho_adaptation=False, data_dir='../data'):
    """Plot state and cost metrics from saved files"""
    data_dir = Path(data_dir)
    
    # Load data from files
    costs = np.loadtxt(data_dir / 'costs' / f'costs{suffix}.txt')
    violations = np.loadtxt(data_dir / 'violations' / f'violations{suffix}.txt')
    
    plt.figure(figsize=(15, 5))
    
    # Plot costs
    plt.subplot(121)
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
    plt.plot(violations[:, 1], label='State Violation')
    plt.xlabel('MPC Step')
    plt.ylabel('State Constraint Violation')
    plt.title('State Violations')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_all_metrics(suffix, use_rho_adaptation=False, data_dir='../data', dt=0.02):
    """Plot all metrics from saved files"""
    data_dir = Path(data_dir)
    
    # Load all data from files
    iterations = np.loadtxt(data_dir / 'iterations' / f'traj{suffix}.txt')
    violations = np.loadtxt(data_dir / 'violations' / f'violations{suffix}.txt')
    trajectory_costs = np.loadtxt(data_dir / 'trajectory_costs' / f'traj{suffix}.txt')
    control_efforts = np.loadtxt(data_dir / 'control_efforts' / f'traj{suffix}.txt')
    
    if use_rho_adaptation:
        rho_history = np.loadtxt(data_dir / 'rho_history' / f'traj{suffix}.txt')
    else:
        rho_history = None
    
    plt.figure(figsize=(20, 10))
    
    # Plot tracking error/trajectory costs
    plt.subplot(231)
    plt.plot(np.arange(len(trajectory_costs))*dt, trajectory_costs)
    plt.xlabel('Time [s]')
    plt.ylabel('Trajectory Cost')
    plt.title('Tracking Cost over Time')
    plt.grid(True)
    
    # Plot violations
    plt.subplot(232)
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
                f"Avg traj cost: {np.mean(trajectory_costs):.4f}\n" +
                f"Avg control effort: {np.mean(control_efforts):.4f}",
                horizontalalignment='center',
                verticalalignment='center',
                transform=plt.gca().transAxes)
        plt.axis('off')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_comparisons(data_dir='../data', traj_type='full', compare_type='normal'):
    """Load saved data and create comparison plots"""
    try:
        data_dir = Path(data_dir)
        print("\nLoading metrics for comparison...")
        
        # Build suffixes exactly as used in hover.py/traj.py
        if compare_type == 'wind':
            adaptive_suffix = f'_adaptive_wind_{traj_type}'
            fixed_suffix = f'_normal_wind_{traj_type}'
            title_suffix = 'with Wind'
        else:
            adaptive_suffix = f'_adaptive_{traj_type}'
            fixed_suffix = f'_normal_{traj_type}'
            title_suffix = ''
            
        # Load data using exact same paths as hover.py/traj.py
        adaptive_costs = np.loadtxt(data_dir / 'costs' / f"costs{adaptive_suffix}.txt")
        adaptive_violations = np.loadtxt(data_dir / 'violations' / f"violations{adaptive_suffix}.txt")
        adaptive_traj_costs = np.loadtxt(data_dir / 'trajectory_costs' / f"traj{adaptive_suffix}.txt")
        adaptive_control = np.loadtxt(data_dir / 'control_efforts' / f"traj{adaptive_suffix}.txt")
        adaptive_iterations = np.loadtxt(data_dir / 'iterations' / f"traj{adaptive_suffix}.txt")
        
        # Load fixed data
        fixed_costs = np.loadtxt(data_dir / 'costs' / f"costs{fixed_suffix}.txt")
        fixed_violations = np.loadtxt(data_dir / 'violations' / f"violations{fixed_suffix}.txt")
        fixed_traj_costs = np.loadtxt(data_dir / 'trajectory_costs' / f"traj{fixed_suffix}.txt")
        fixed_control = np.loadtxt(data_dir / 'control_efforts' / f"traj{fixed_suffix}.txt")
        fixed_iterations = np.loadtxt(data_dir / 'iterations' / f"traj{fixed_suffix}.txt")
        
        # Create figure with extra space on right for stats
        plt.figure(figsize=(22, 10))  # Made figure wider to accommodate stats
        
        # State Costs
        plt.subplot(231)
        plt.plot(fixed_costs[:, 0], 'b-o', label='Fixed', alpha=0.5, linewidth=2, markersize=4, markevery=20)
        plt.plot(adaptive_costs[:, 0], 'r-^', label='Adaptive', alpha=0.8, linewidth=2.5, markersize=6, markevery=20)
        plt.xlabel('MPC Step')
        
        #plt.title(f'State Costs Comparison ({title_suffix})')
        plt.title('State Costs Comparison')
        plt.legend()
        plt.grid(True)
        
        # Input Costs
        plt.subplot(232)
        plt.plot(fixed_costs[:, 1], 'b-s', label='Fixed', alpha=0.5, linewidth=2, markersize=4, markevery=20)
        plt.plot(adaptive_costs[:, 1], 'r-d', label='Adaptive', alpha=0.8, linewidth=2.5, markersize=6, markevery=20)
        plt.xlabel('MPC Step')
        
        plt.title('Input Costs Comparison')
        plt.legend()
        plt.grid(True)
        
        # Input Violations
        plt.subplot(233)
        plt.plot(fixed_violations[:, 0], 'b-*', label='Fixed', alpha=0.5, linewidth=2, markersize=4, markevery=20)
        plt.plot(adaptive_violations[:, 0], 'r-p', label='Adaptive', alpha=0.8, linewidth=2.5, markersize=6, markevery=20)
        plt.xlabel('MPC Step')
        
        plt.title('Input Violations')
        plt.legend()
        plt.grid(True)

        # State Violations (with adjusted scale)
        plt.subplot(234)
        plt.plot(fixed_violations[:, 1], 'b-*', label='Fixed', alpha=0.5, linewidth=2, markersize=4, markevery=20)
        plt.plot(adaptive_violations[:, 1], 'r-p', label='Adaptive', alpha=0.8, linewidth=2.5, markersize=6, markevery=20)
        plt.xlabel('MPC Step')
        
        plt.title('State Violations (×10⁻³)')
        plt.legend()
        plt.grid(True)

        #Iterations
        plt.subplot(235)
        plt.plot(fixed_iterations, 'b-*', label='Fixed', alpha=0.5, linewidth=2, markersize=4, markevery=20)
        plt.plot(adaptive_iterations, 'r-p', label='Adaptive', alpha=0.8, linewidth=2.5, markersize=6, markevery=20)
        plt.xlabel('MPC Step')
        
        plt.title('Iterations')
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
            f'Total Iterations:\n'
            f'  Fixed: {np.sum(fixed_iterations)}\n'
            f'  Adaptive: {np.sum(adaptive_iterations)}\n\n'

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

def plot_hover_iterations_comparison():
    """Compare iterations between different hover control approaches"""
    data_dir = Path('../data/iterations')
    
    # Load the data
    normal = np.loadtxt(data_dir / 'traj_normal_hover.txt')
    adaptive = np.loadtxt(data_dir / 'traj_adaptive_hover.txt')
    recompute = np.loadtxt(data_dir / 'traj_adaptive_recache_hover.txt')
    
    # Create plot
    plt.figure(figsize=(10, 6), dpi=100)
    t = np.arange(len(normal)) * 0.02  # assuming dt = 0.02
    
    plt.plot(t, normal, 'b-', label='Fixed ρ', linewidth=2)
    plt.plot(t, adaptive, 'r-', label='Adaptive ρ', linewidth=2)
    plt.plot(t, recompute, 'g-', label='Adaptive ρ (recompute)', linewidth=2)
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Iterations', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.title('ADMM Iterations Comparison for Hover Control', fontsize=14)
    
    plt.tight_layout()
    plt.show()