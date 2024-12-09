import json
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def calculate_response_variances(scores):
    """Calculate variance for each response's set of scores"""
    return [np.var(response_scores) for response_scores in scores]

def analyze_reasoning_effect():
    """
    Analyze how reasoning affects score variance across different models
    """
    # Load results
    exp1c_path = Path("results/exp1c/combined_results.json")
    exp1c_reasoning_path = Path("results/exp1c_reasoning/combined_results_reasoning.json")

    with open(exp1c_path, 'r') as f:
        no_reasoning_results = json.load(f)
    with open(exp1c_reasoning_path, 'r') as f:
        reasoning_results = json.load(f)

    # Models and temperatures to analyze
    models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
    temperatures = ['0.3', '0.5', '1.0']

    # Collect variance data
    data = []
    for temp in temperatures:
        for model in models:
            # Get variances for each condition
            no_reasoning_variances = calculate_response_variances(
                no_reasoning_results['results'][temp][model]['raw_scores']
            )
            reasoning_variances = calculate_response_variances(
                reasoning_results['results'][temp][model]['raw_scores']
            )

            # Calculate variance ratio (reasoning/no_reasoning) for each response
            variance_ratios = []
            for no_r_var, r_var in zip(no_reasoning_variances, reasoning_variances):
                if no_r_var > 0:  # Avoid division by zero
                    variance_ratios.append(r_var / no_r_var)

            # Add to dataset
            for ratio in variance_ratios:
                data.append({
                    'Model': model,
                    'Temperature': temp,
                    'Variance_Ratio': ratio
                })

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Perform Kruskal-Wallis H-test for each temperature
    print("\nKruskal-Wallis H-test results by temperature:")
    for temp in temperatures:
        temp_data = df[df['Temperature'] == temp]
        groups = [group['Variance_Ratio'].values for name, group in temp_data.groupby('Model')]
        h_stat, p_val = stats.kruskal(*groups)
        print(f"\nTemperature {temp}:")
        print(f"H-statistic: {h_stat:.4f}")
        print(f"p-value: {p_val:.4f}")
        
        # Perform post-hoc analysis if p < 0.1
        if p_val < 0.1:
            print("\nPost-hoc analysis (Mann-Whitney U tests):")
            # Perform pairwise comparisons
            for i, model1 in enumerate(models):
                for model2 in models[i+1:]:
                    group1 = temp_data[temp_data['Model'] == model1]['Variance_Ratio']
                    group2 = temp_data[temp_data['Model'] == model2]['Variance_Ratio']
                    stat, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                    
                    # Calculate effect size (r = Z / sqrt(N))
                    n1, n2 = len(group1), len(group2)
                    z_score = stat - (n1 * n2 / 2)
                    z_score /= np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
                    effect_size = abs(z_score) / np.sqrt(n1 + n2)
                    
                    # Get medians for interpretation
                    median1 = np.median(group1)
                    median2 = np.median(group2)
                    
                    print(f"\n{model1} vs {model2}:")
                    print(f"p-value: {p:.4f}")
                    print(f"Effect size (r): {effect_size:.4f}")
                    print(f"Median ratio {model1}: {median1:.4f}")
                    print(f"Median ratio {model2}: {median2:.4f}")
                    
                    # Interpret effect size
                    if effect_size >= 0.5:
                        effect_magnitude = "large"
                    elif effect_size >= 0.3:
                        effect_magnitude = "medium"
                    elif effect_size >= 0.1:
                        effect_magnitude = "small"
                    else:
                        effect_magnitude = "negligible"
                    
                    # Interpret direction
                    if median1 > median2:
                        direction = f"{model1} more affected"
                    else:
                        direction = f"{model2} more affected"
                    
                    if p < 0.1:
                        print(f"Significant difference: {direction} ({effect_magnitude} effect)")

    # Create violin plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x='Model', y='Variance_Ratio', hue='Temperature')
    plt.title('Distribution of Variance Ratios (Reasoning/No-Reasoning) by Model')
    plt.ylabel('Variance Ratio')
    plt.xticks(rotation=45)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)  # Reference line at ratio=1
    plt.tight_layout()
    plt.savefig('results/reasoning_effect_comparison.png')
    plt.close()

    # Print summary statistics
    print("\nSummary statistics (median variance ratios):")
    summary = df.groupby(['Model', 'Temperature'])['Variance_Ratio'].median()
    print(summary)

if __name__ == "__main__":
    analyze_reasoning_effect() 