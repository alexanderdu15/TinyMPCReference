import numpy as np
import matplotlib.pyplot as plt

def plot_iteration_comparison():
    # Create figure with 3 subplots vertically aligned
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Load data
    base_path = "/Users/ishaanmahajan/A2R_Research/workspace/TinyMPCReference/examples/data/paper_plots/"
    normal = np.loadtxt(f"{base_path}iterations_normal_hover.txt")
    adapt_1 = np.loadtxt(f"{base_path}iterations_adaptive_freq_1_hover.txt")
    adapt_recache_1 = np.loadtxt(f"{base_path}iterations_adaptive_freq_1_recache_freq_1_hover.txt")
    adapt_5 = np.loadtxt(f"{base_path}iterations_adaptive_freq_5_hover.txt")
    adapt_recache_5 = np.loadtxt(f"{base_path}iterations_adaptive_freq_5_recache_freq_5_hover.txt")
    adapt_10 = np.loadtxt(f"{base_path}iterations_adaptive_freq_10_hover.txt")
    adapt_recache_10 = np.loadtxt(f"{base_path}iterations_adaptive_freq_10_recache_freq_10_hover.txt")
    
    # Calculate total iterations for each method
    total_normal = np.sum(normal)
    
    # Plot 1: Every step updates
    ax1.plot(normal, 'k-', label='Baseline', linewidth=2, alpha=0.5)
    ax1.plot(adapt_1, 'b-', label='Adapt Only', linewidth=2)
    ax1.plot(adapt_recache_1, 'r--', label='Adapt + Recache', linewidth=2)
    ax1.set_title('Updates Every Step')
    ax1.set_ylabel('Iterations')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add textbox for total iterations
    textstr = '\n'.join((
        'Total Iterations:',
        f'Baseline: {total_normal:,.0f}',
        f'Adapt Only: {np.sum(adapt_1):,.0f}',
        f'Adapt + Recache: {np.sum(adapt_recache_1):,.0f}'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=props)
    
    # Plot 2: Every 5 steps
    ax2.plot(normal, 'k-', label='Baseline', linewidth=2, alpha=0.5)
    ax2.plot(adapt_5, 'b-', label='Adapt Only', linewidth=2)
    ax2.plot(adapt_recache_5, 'r--', label='Adapt + Recache', linewidth=2)
    ax2.set_title('Updates Every 5 Steps')
    ax2.set_ylabel('Iterations')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add textbox for total iterations
    textstr = '\n'.join((
        'Total Iterations:',
        f'Baseline: {total_normal:,.0f}',
        f'Adapt Only: {np.sum(adapt_5):,.0f}',
        f'Adapt + Recache: {np.sum(adapt_recache_5):,.0f}'))
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=props)
    
    # Plot 3: Every 10 steps
    ax3.plot(normal, 'k-', label='Baseline', linewidth=2, alpha=0.5)
    ax3.plot(adapt_10, 'b-', label='Adapt Only', linewidth=2)
    ax3.plot(adapt_recache_10, 'r--', label='Adapt + Recache', linewidth=2)
    ax3.set_title('Updates Every 10 Steps')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Iterations')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Add textbox for total iterations
    textstr = '\n'.join((
        'Total Iterations:',
        f'Baseline: {total_normal:,.0f}',
        f'Adapt Only: {np.sum(adapt_10):,.0f}',
        f'Adapt + Recache: {np.sum(adapt_recache_10):,.0f}'))
    ax3.text(0.02, 0.98, textstr, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', bbox=props)
    
    # Set y-axis limits to better show the differences
    for ax in [ax1, ax2, ax3]:
        ax.set_ylim(0, 500)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"{base_path}iteration_comparison.pdf", bbox_inches='tight', dpi=300)
    plt.show()

# Run the plotting function
plot_iteration_comparison()