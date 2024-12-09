import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_model_temperatures(exp_name, output_suffix=""):
    """
    Create plots showing centered score distributions for each model across different temperatures
    
    Args:
        exp_name: Name of experiment folder (e.g., 'exp1c' or 'exp1c_reasoning')
        output_suffix: Suffix to add to output filename
    """
    # Load combined results
    exp_path = Path(f"results/{exp_name}/combined_results{output_suffix}.json")
    with open(exp_path, 'r') as f:
        results = json.load(f)

    # Create figure with 3 subplots (one for each model)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    title_suffix = " (with Reasoning)" if "reasoning" in exp_name else ""
    fig.suptitle(f'Centered Score Distribution by Model and Temperature{title_suffix}', 
                fontsize=16, y=1.05)

    # Colors for different temperatures
    colors = {
        '0.3': '#2ecc71',  # green
        '0.5': '#3498db',  # blue
        '1.0': '#e74c3c'   # red
    }

    models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
    temperatures = ['0.3', '0.5', '1.0']

    # Process data for each model
    for model_idx, model in enumerate(models):
        ax = axes[model_idx]
        model_data = []
        
        # Collect data for each temperature
        for temp in temperatures:
            # Get raw scores and center them by group
            for response_scores in results['results'][temp][model]['raw_scores']:
                mean_score = np.mean(response_scores)
                centered_scores = [score - mean_score for score in response_scores]
                
                # Create DataFrame for these centered scores
                temp_df = pd.DataFrame({
                    'Centered Score': centered_scores,
                    'Temperature': temp
                })
                model_data.append(temp_df)
        
        # Combine all temperatures for this model
        model_df = pd.concat(model_data)
        
        # Create density plot for each temperature
        for temp in temperatures:
            temp_data = model_df[model_df['Temperature'] == temp]['Centered Score']
            sns.kdeplot(
                data=temp_data,
                ax=ax,
                label=f'Temp {temp}',
                color=colors[temp],
                fill=True,
                alpha=0.3
            )
        
        # Customize subplot
        ax.set_title(f'{model}')
        ax.set_xlabel('Centered Score')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set consistent x-axis limits
        ax.set_xlim(-5, 5)  # Adjusted for centered scores

    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_path = Path(f"results/{exp_name}/temperature_comparison_centered{output_suffix}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    # Plot both regular and reasoning results
    plot_model_temperatures("exp1c")
    plot_model_temperatures("exp1c_reasoning", "_reasoning") 