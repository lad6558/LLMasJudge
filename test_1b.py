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
    
    # After existing tests, add:
    print("\n" + "="*80)
    print("Testing interaction between model size and temperature sensitivity:")
    
    # Calculate temperature sensitivity (correlation with temperature) for each model
    model_sensitivities = {}
    for model in models:
        model_data = df[df['Model'] == model]
        
        # Get correlation coefficient as measure of temperature sensitivity
        correlation, _ = stats.spearmanr(model_data['Temperature'], 
                                       model_data['Variance'])
        model_sensitivities[model] = correlation
    
    print("\nTemperature sensitivity (Spearman correlation):")
    for model, sensitivity in model_sensitivities.items():
        print(f"{model}: {sensitivity:.4f}")
    
    # Compare slopes of variance increase
    print("\nComparing rates of variance increase:")
    for temp in temperatures:
        print(f"\nTemperature {temp}:")
        temp_data = df[df['Temperature'] == temp]
        
        # Get variances for each model
        groups = [group['Variance'].values for name, group in 
                 temp_data.groupby('Model')]
        
        if all(len(g) > 0 for g in groups):
            # Check if all values are identical
            all_values = np.concatenate(groups)
            if np.all(all_values == all_values[0]):
                print("All models produced identical variances at this temperature")
                print(f"Variance value: {all_values[0]:.4f}")
                continue
                
            # Proceed with Kruskal-Wallis test if values differ
            h_stat, p_value = stats.kruskal(*groups)
            print(f"Kruskal-Wallis test - p-value: {p_value:.4f}")
            
            # If significant, do pairwise comparisons
            if p_value < 0.05:
                print("Pairwise Mann-Whitney U tests:")
                # Compare GPT-4o vs GPT-4o-mini
                g1 = temp_data[temp_data['Model'] == 'gpt-4o']['Variance']
                g2 = temp_data[temp_data['Model'] == 'gpt-4o-mini']['Variance']
                _, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                print(f"GPT-4o vs GPT-4o-mini: p={p:.4f}")
                
                # Compare GPT-4o-mini vs GPT-3.5-turbo
                g2 = temp_data[temp_data['Model'] == 'gpt-4o-mini']['Variance']
                g3 = temp_data[temp_data['Model'] == 'gpt-3.5-turbo']['Variance']
                _, p = stats.mannwhitneyu(g2, g3, alternative='two-sided')
                print(f"GPT-4o-mini vs GPT-3.5-turbo: p={p:.4f}")
    
    # Create interaction plot
    plt.figure(figsize=(12, 8))
    for model in models:
        model_data = df[df['Model'] == model].groupby('Temperature')['Variance'].mean()
        plt.plot(model_data.index, model_data.values, marker='o', label=model)
    
    plt.xlabel('Temperature')
    plt.ylabel('Mean Variance')
    plt.title('Temperature Sensitivity by Model')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/model_temperature_sensitivity.png')
    plt.close()
    
    # Test for differences in variance increase rate
    print("\nTesting differences in variance increase rate:")
    # Calculate variance increase rate (slope) for each model
    slopes = {}
    for model in models:
        model_data = df[df['Model'] == model]
        mean_variances = model_data.groupby('Temperature')['Variance'].mean()
        # Use linear regression to get slope
        slope, _, _, _, _ = stats.linregress(temperatures, mean_variances)
        slopes[model] = slope
    
    print("\nVariance increase rates (slopes):")
    for model, slope in slopes.items():
        print(f"{model}: {slope:.4f}")
    
    # Identify which model is most/least affected by temperature
    most_sensitive = max(slopes.items(), key=lambda x: x[1])[0]
    least_sensitive = min(slopes.items(), key=lambda x: x[1])[0]
    
    print(f"\nMost temperature-sensitive model: {most_sensitive}")
    print(f"Least temperature-sensitive model: {least_sensitive}")

if __name__ == "__main__":
    test_temperature_variance_hypothesis() 