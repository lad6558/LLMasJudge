import json
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def test_temperature_variance_hypothesis():
    """
    Test hypothesis: Higher temperature leads to higher variance in judgments
    Using data from experiment 1b across different models and all temperatures
    """
    # Load results
    exp1b_path = Path("results/exp1b/combined_results.json")
    
    with open(exp1b_path, 'r') as f:
        exp1b_results = json.load(f)
    
    # Extract data for each temperature and model
    temperatures = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
    
    # Prepare data for analysis
    plot_data = []
    for temp in temperatures:
        temp_str = str(temp)
        for model in models:
            try:
                variances = exp1b_results['results'][temp_str][model]['individual_variances']
                avg_variance = exp1b_results['results'][temp_str][model]['average_variance']
                
                # Add each variance with its temperature
                for variance in variances:
                    plot_data.append({
                        'Temperature': temp,
                        'Model': model,
                        'Variance': variance
                    })
                
            except KeyError:
                print(f"Warning: No data for {model} at temperature {temp}")
    
    # Convert to DataFrame
    df = pd.DataFrame(plot_data)
    
    # Create violin plot
    plt.figure(figsize=(15, 8))
    sns.violinplot(x='Temperature', y='Variance', hue='Model', data=df)
    plt.title('Score Variance Distribution by Temperature and Model')
    plt.ylabel('Variance in Scores')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/temperature_variance_comparison.png')
    plt.close()
    
    # Create line plot with error bands
    plt.figure(figsize=(15, 8))
    sns.lineplot(data=df, x='Temperature', y='Variance', hue='Model', err_style='band')
    plt.title('Mean Variance by Temperature with 95% Confidence Intervals')
    plt.ylabel('Variance in Scores')
    plt.tight_layout()
    plt.savefig('results/temperature_variance_trend.png')
    plt.close()
    
    # Statistical tests
    print("\nTemperature vs Variance Analysis:")
    
    # Spearman correlation for each model
    for model in models:
        model_data = df[df['Model'] == model]
        correlation, p_value = stats.spearmanr(model_data['Temperature'], 
                                             model_data['Variance'])
        print(f"\n{model}:")
        print(f"Spearman correlation: {correlation:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        # Print mean variances at each temperature
        print("\nMean variances by temperature:")
        for temp in temperatures:
            mean_var = model_data[model_data['Temperature'] == temp]['Variance'].mean()
            print(f"Temperature {temp}: {mean_var:.4f}")
    
    # Kruskal-Wallis H-test for each model
    print("\nKruskal-Wallis H-test results:")
    for model in models:
        model_data = df[df['Model'] == model]
        groups = [group['Variance'].values for name, group in 
                 model_data.groupby('Temperature')]
        
        h_stat, p_value = stats.kruskal(*groups)
        print(f"\n{model}:")
        print(f"H-statistic: {h_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        # If significant, perform pairwise Mann-Whitney U tests with Holm's correction
        if p_value < 0.05:
            print("\nPairwise Mann-Whitney U tests with Holm's correction:")
            # Store all pairwise comparisons
            pairwise_tests = []
            for i, temp1 in enumerate(temperatures[:-1]):
                for temp2 in temperatures[i+1:]:
                    group1 = model_data[model_data['Temperature'] == temp1]['Variance']
                    group2 = model_data[model_data['Temperature'] == temp2]['Variance']
                    stat, p = stats.mannwhitneyu(group1, group2, alternative='less')
                    pairwise_tests.append((p, f"{temp1} vs {temp2}"))
            
            # Apply Holm's correction
            pairwise_tests.sort()  # Sort by p-value
            m = len(pairwise_tests)
            for i, (p, comparison) in enumerate(pairwise_tests):
                adjusted_p = p * (m - i)  # Holm's correction
                adjusted_p = min(adjusted_p, 1.0)  # Cap at 1.0
                print(f"Temperature {comparison} - adjusted p-value: {adjusted_p:.4f}")

if __name__ == "__main__":
    test_temperature_variance_hypothesis() 