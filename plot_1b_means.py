import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_mean_variances():
    # Load results
    results_dir = Path("results/exp1b")
    combined_file = results_dir / "combined_results.json"
    
    with open(combined_file, 'r') as f:
        data = json.load(f)
    
    # Setup plot style
    plt.style.use('seaborn')
    colors = {
        'gpt-3.5-turbo': '#FF6B6B',  # Red
        'gpt-4o-mini': '#4ECDC4',    # Teal
        'gpt-4o': '#45B7D1'          # Blue
    }
    
    # Extract temperatures and sort them
    temperatures = sorted([float(t) for t in data['results'].keys()])
    models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
    
    plt.figure(figsize=(10, 6))
    
    # Plot for each model
    for model in models:
        mean_variances = []
        for temp in temperatures:
            try:
                variance = data['results'][str(temp)][model]['average_variance']
                mean_variances.append(variance)
            except KeyError:
                mean_variances.append(np.nan)
        
        plt.plot(temperatures, mean_variances, 
                marker='o', 
                label=model, 
                color=colors[model], 
                linewidth=2, 
                markersize=6)
    
    # Customize plot
    plt.title('Mean Variance by Temperature and Model (Experiment 1b)', fontsize=12)
    plt.xlabel('Temperature', fontsize=10)
    plt.ylabel('Mean Variance', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=9)
    
    # Save plot
    plot_file = results_dir / "mean_variances.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_file}")
    plt.close()

if __name__ == "__main__":
    plot_mean_variances() 