import numpy as np
import matplotlib.pyplot as plt

def plot_cumulative_iterations():
    # Load data
    base_path = "/Users/ishaanmahajan/A2R_Research/workspace/TinyMPCReference/examples/data/paper_plots/"
    normal = np.loadtxt(f"{base_path}iterations_normal_hover.txt")
    adapt_10 = np.loadtxt(f"{base_path}iterations_adaptive_freq_10_hover.txt")
    adapt_recache_10 = np.loadtxt(f"{base_path}iterations_adaptive_freq_10_recache_freq_10_hover.txt")
    
    # Calculate cumulative sums
    cum_normal = np.cumsum(normal)
    cum_adapt = np.cumsum(adapt_10)
    cum_recache = np.cumsum(adapt_recache_10)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot cumulative iterations
    plt.plot(cum_normal, 'k-', label='Baseline', linewidth=2, alpha=0.7)
    plt.plot(cum_adapt, 'b-', label='Adapt (10 steps)', linewidth=2)
    plt.plot(cum_recache, 'r--', label='Adapt + Recache (10 steps)', linewidth=2)
    
    # Add final iteration counts in text box
    textstr = '\n'.join((
        'Total Iterations:',
        f'Baseline: {cum_normal[-1]:,.0f}',
        f'Adapt (10): {cum_adapt[-1]:,.0f}',
        f'Adapt + Recache (10): {cum_recache[-1]:,.0f}'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Customize plot
    plt.title('Cumulative ADMM Iterations Over Time', fontsize=14)
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Iterations')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    
    # Add percentage improvement annotations
    pct_improve_adapt = ((cum_normal[-1] - cum_adapt[-1]) / cum_normal[-1]) * 100
    pct_improve_recache = ((cum_normal[-1] - cum_recache[-1]) / cum_normal[-1]) * 100
    
    textstr_pct = '\n'.join((
        'Improvement vs Baseline:',
        f'Adapt: {pct_improve_adapt:.1f}%',
        f'Adapt + Recache: {pct_improve_recache:.1f}%'))
    plt.text(0.98, 0.02, textstr_pct, transform=plt.gca().transAxes, fontsize=10,
             horizontalalignment='right', verticalalignment='bottom', bbox=props)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"{base_path}cumulative_iterations.pdf", bbox_inches='tight', dpi=300)
    plt.show()

# Run the plotting function
plot_cumulative_iterations()