import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_reasoning_variances():
    """
    Create bar plot comparing variances across temperatures for:
    - Reasoning before scoring (exp2a)
    - Reasoning after scoring (exp2b)
    - No reasoning (exp1b)
    """
    # Load results
    exp2a_path = Path("results/exp2a/combined_results.json")
    exp2b_path = Path("results/exp2b/combined_results.json")
    # No reasoning baseline
    exp1b_path = Path("results/exp1b/combined_results.json")

    with open(exp2a_path, 'r') as f:
        exp2a_results = json.load(f)
    with open(exp2b_path, 'r') as f:
        exp2b_results = json.load(f)
    with open(exp1b_path, 'r') as f:
        exp1b_results = json.load(f)

    # Extract data
    temperatures = ['0.3', '0.5', '1.0']
    data = []

    for temp in temperatures:
        # Get variances for each condition
        reasoning_before = exp2a_results['results'][temp]['individual_variances']
        reasoning_after = exp2b_results['results'][temp]['individual_variances']
        no_reasoning = exp1b_results['results'][temp]['gpt-4o-mini']['individual_variances']

        # Add to data list
        for variance in reasoning_before:
            data.append({'Temperature': float(temp),
                        'Condition': 'Reasoning Before', 'Variance': variance})
        for variance in reasoning_after:
            data.append({'Temperature': float(temp),
                        'Condition': 'Reasoning After', 'Variance': variance})
        for variance in no_reasoning:
            data.append({'Temperature': float(temp),
                        'Condition': 'No Reasoning', 'Variance': variance})

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Create plot
    plt.figure(figsize=(12, 6))

    # Create grouped bar plot
    sns.barplot(
        data=df,
        x='Temperature',
        y='Variance',
        hue='Condition',
        palette={
            'Reasoning Before': '#2ecc71',  # green
            'Reasoning After': '#3498db',   # blue
            'No Reasoning': '#e74c3c'       # red
        },
        capsize=0.05,
        errwidth=2,
        ci=95  # 95% confidence intervals
    )

    # Customize plot
    plt.title('Score Variance by Temperature and Reasoning Condition', pad=20)
    plt.xlabel('Temperature')
    plt.ylabel('Variance in Scores')

    # Add grid
    plt.grid(True, axis='y', alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save plot
    output_path = Path("results/variance_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Print statistical summary
    print("\nMean Variances by Condition and Temperature:")
    summary = df.groupby(['Temperature', 'Condition'])[
        'Variance'].agg(['mean', 'std']).round(4)
    print(summary)

    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    plot_reasoning_variances()
