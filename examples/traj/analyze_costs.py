import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt

def plot_trajectory_cost_analysis(data_dir='../data', traj_type='full'):
    """Plot percentage difference in costs between adaptive and fixed"""
    try:
        data_dir = Path(data_dir)
        
        # Load costs
        adaptive_costs = np.loadtxt(data_dir / 'costs' / f"costs_adaptive_{traj_type}.txt")
        fixed_costs = np.loadtxt(data_dir / 'costs' / f"costs_normal_{traj_type}.txt")
        
        # Calculate average costs
        avg_adaptive = [np.mean(adaptive_costs[:, i]) for i in range(3)]
        avg_fixed = [np.mean(fixed_costs[:, i]) for i in range(3)]
        
        # Calculate percentage differences
        diff_percent = [(a - f) / f * 100 for a, f in zip(avg_adaptive, avg_fixed)]
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        x = np.arange(3)
        plt.bar(x, diff_percent, color=['blue' if d < 0 else 'red' for d in diff_percent])
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xticks(x, ['State', 'Input', 'Total'])
        plt.title('Cost Difference: Adaptive vs Fixed (%)')
        plt.ylabel('Percent Difference')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(diff_percent):
            plt.text(i, v + np.sign(v)*0.5, f'{v:+.1f}%', 
                    ha='center', va='bottom' if v > 0 else 'top')
        
       
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error analyzing costs: {e}")

if __name__ == "__main__":
    plot_trajectory_cost_analysis()