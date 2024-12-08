import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

def load_and_plot_results():
    # Load results from exp1c
    results_dir = Path("results/exp1c")
    combined_file = results_dir / "combined_results.json"
    
    with open(combined_file, 'r') as f:
        data = json.load(f)
    
    # Setup plot style
    plt.style.use('seaborn')
    colors = ['#FF9999', '#66B2FF', '#99FF99']  # Red, Blue, Green for temperatures
    temperatures = ['0.3', '0.5', '1.0']
    models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Sampling Distribution of Variances by Model and Temperature (Experiment 1c)', fontsize=14)
    
    # Plot for each model
    for idx, model in enumerate(models):
        ax = axes[idx]
        
        # Collect variances for each temperature
        for temp_idx, temp in enumerate(temperatures):
            variances = data['results'][temp][model]['individual_variances']
            # Center the variances around 0
            centered_variances = np.array(variances) - np.mean(variances)
            
            # Create histogram
            sns.histplot(
                centered_variances,
                ax=ax,
                color=colors[temp_idx],
                alpha=0.5,
                label=f'Temp {temp}',
                bins=20,
                stat='count'
            )
        
        # Customize plot
        ax.set_title(f'{model}')
        ax.set_xlabel('Centered Variance')
        ax.set_ylabel('Frequency')
        ax.set_xlim(-5, 5)
        ax.legend()
        
    plt.tight_layout()
    
    # Save plot
    plot_file = results_dir / "variance_distributions.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_file}")
    plt.close()

if __name__ == "__main__":
    load_and_plot_results() 