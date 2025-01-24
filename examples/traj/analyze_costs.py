import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt

def plot_trajectory_cost_analysis(data_dir='../../data', traj_type='full'):
    """Analyze and plot detailed cost differences for trajectory data"""
    try:
        data_dir = Path(data_dir)
        
        # Load costs
        adaptive_costs = np.loadtxt(data_dir / 'costs' / f"costs_adaptive_{traj_type}.txt")
        fixed_costs = np.loadtxt(data_dir / 'costs' / f"costs_normal_{traj_type}.txt")
        
        plt.figure(figsize=(15, 12))
        
        # Plot 1: Absolute Difference Over Time
        plt.subplot(2, 2, 1)
        time = np.arange(len(adaptive_costs))
        differences = adaptive_costs - fixed_costs
        plt.plot(time, differences[:, 0], label='State Cost Diff', alpha=0.8)
        plt.plot(time, differences[:, 1], label='Input Cost Diff', alpha=0.8)
        plt.plot(time, differences[:, 2], label='Total Cost Diff', alpha=0.8)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Time Step')
        plt.ylabel('Cost Difference (Adaptive - Fixed)')
        plt.title('Absolute Cost Differences Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Relative Difference Over Time (%)
        plt.subplot(2, 2, 2)
        rel_diff = (adaptive_costs - fixed_costs) / fixed_costs * 100
        plt.plot(time, rel_diff[:, 0], label='State Cost', alpha=0.8)
        plt.plot(time, rel_diff[:, 1], label='Input Cost', alpha=0.8)
        plt.plot(time, rel_diff[:, 2], label='Total Cost', alpha=0.8)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Time Step')
        plt.ylabel('Relative Difference (%)')
        plt.title('Relative Cost Differences Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative Costs
        plt.subplot(2, 2, 3)
        cum_adaptive = np.cumsum(adaptive_costs[:, 2])
        cum_fixed = np.cumsum(fixed_costs[:, 2])
        plt.plot(time, cum_adaptive, label='Adaptive', alpha=0.8)
        plt.plot(time, cum_fixed, label='Fixed', alpha=0.8)
        plt.fill_between(time, cum_adaptive, cum_fixed, alpha=0.2)
        plt.xlabel('Time Step')
        plt.ylabel('Cumulative Total Cost')
        plt.title('Cumulative Cost Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Moving Average of Difference
        plt.subplot(2, 2, 4)
        window = 20  # Window size for moving average
        moving_avg = np.zeros_like(differences)
        for i in range(3):
            moving_avg[:, i] = np.convolve(differences[:, i], 
                                         np.ones(window)/window, 
                                         mode='valid')
        valid_time = time[window-1:]
        plt.plot(valid_time, moving_avg[window-1:, 0], label='State Cost', alpha=0.8)
        plt.plot(valid_time, moving_avg[window-1:, 1], label='Input Cost', alpha=0.8)
        plt.plot(valid_time, moving_avg[window-1:, 2], label='Total Cost', alpha=0.8)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Time Step')
        plt.ylabel('Moving Average of Difference')
        plt.title(f'Moving Average of Cost Differences (Window={window})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        avg_diff = np.mean(differences, axis=0)
        std_diff = np.std(differences, axis=0)
        stats_text = (
            f'Average Differences (Adaptive - Fixed):\n'
            f'State: {avg_diff[0]:+.3f} ± {std_diff[0]:.3f}\n'
            f'Input: {avg_diff[1]:+.3f} ± {std_diff[1]:.3f}\n'
            f'Total: {avg_diff[2]:+.3f} ± {std_diff[2]:.3f}\n\n'
            f'Cumulative Cost Difference: {(cum_adaptive[-1] - cum_fixed[-1]):+.3f}'
        )
        plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error analyzing costs: {e}")

if __name__ == "__main__":
    plot_trajectory_cost_analysis()