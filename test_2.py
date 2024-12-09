import json
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def test_reasoning_variance_hypothesis():
    """
    Test hypothesis: var(reasoning_before) > var(reasoning_after) > var(no_reasoning)
    Using data from experiments 1, 2a and 2b
    """
    # Load results
    exp2a_path = Path("results/exp2a/combined_results.json")
    exp2b_path = Path("results/exp2b/combined_results.json")

    with open(exp2a_path, 'r') as f:
        exp2a_results = json.load(f)
    with open(exp2b_path, 'r') as f:
        exp2b_results = json.load(f)

    # Extract variances for each temperature
    temperatures = ['0.3', '0.5', '1.0']

    reasoning_before_variances = []
    reasoning_after_variances = []
    no_reasoning_variances = []

    # Get variances from exp2a and exp2b
    for temp in temperatures:
        reasoning_before_variances.extend(
            exp2a_results['results'][temp]['individual_variances'])
        reasoning_after_variances.extend(
            exp2b_results['results'][temp]['individual_variances'])

        # Load no reasoning data from exp1 for each temperature
        temp_str = str(temp).replace('.', '_')
        exp1_path = Path(f"results/exp1b/gpt_4o_mini_temp_{temp_str}.json")
        if exp1_path.exists():
            with open(exp1_path, 'r') as f:
                exp1_data = json.load(f)
                # Get variances directly from the file
                no_reasoning_variances.extend(
                    exp1_data['individual_variances'])
        else:
            print(f"Warning: No data found at {exp1_path}")

    # Only perform statistical tests if we have data for all conditions
    if not (reasoning_before_variances and reasoning_after_variances and no_reasoning_variances):
        print("\nWarning: Missing data for one or more conditions:")
        print(f"Reasoning before: {len(reasoning_before_variances)} samples")
        print(f"Reasoning after: {len(reasoning_after_variances)} samples")
        print(f"No reasoning: {len(no_reasoning_variances)} samples")
        return

    # Perform statistical tests
    # Levene's test for equality of variances (3-way comparison)
    stat, p_value = stats.levene(
        reasoning_before_variances,
        reasoning_after_variances,
        no_reasoning_variances
    )

    # Create violin plot
    plt.figure(figsize=(10, 6))
    data = {
        'Reasoning Before': reasoning_before_variances,
        'Reasoning After': reasoning_after_variances,
        'No Reasoning': no_reasoning_variances
    }

    # Convert dictionary to DataFrame in long format
    plot_data = []
    for method, values in data.items():
        for value in values:
            plot_data.append({
                'Method': method,
                'Variance': value
            })
    df = pd.DataFrame(plot_data)

    # Create violin plot using the DataFrame
    sns.violinplot(x='Method', y='Variance', data=df)
    plt.title('Distribution of Score Variances by Reasoning Type')
    plt.ylabel('Variance in Scores')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/reasoning_variance_comparison.png')
    plt.close()

    # Print results
    print("\nVariance Comparison Results:")
    print(
        f"Mean variance with reasoning before: {np.mean(reasoning_before_variances):.4f}")
    print(
        f"Mean variance with reasoning after: {np.mean(reasoning_after_variances):.4f}")
    print(
        f"Mean variance with no reasoning: {np.mean(no_reasoning_variances):.4f}")
    print(f"\nLevene's test (all groups):")
    print(f"Statistic: {stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Pairwise comparisons if overall test is significant
    if p_value < 0.10:
        print("\nSignificant differences found - performing pairwise comparisons:")

        # Before vs After
        stat, p = stats.levene(reasoning_before_variances,
                               reasoning_after_variances)
        print(f"\nBefore vs After - p-value: {p:.4f}")

        # Before vs No reasoning
        stat, p = stats.levene(
            reasoning_before_variances, no_reasoning_variances)
        print(f"Before vs No reasoning - p-value: {p:.4f}")

        # After vs No reasoning
        stat, p = stats.levene(reasoning_after_variances,
                               no_reasoning_variances)
        print(f"After vs No reasoning - p-value: {p:.4f}")
    else:
        print("\nNo significant differences in variances between reasoning types")


if __name__ == "__main__":
    test_reasoning_variance_hypothesis()
