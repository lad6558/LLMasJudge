import json
from pathlib import Path
import numpy as np
from scipy import stats
import pandas as pd
from scikit_posthocs import posthoc_nemenyi_friedman
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_experiment_4a():
    # Load results
    results_path = Path("results/exp4a/combined_results.json")
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Extract intermediate scores
    intermediate_scores = results['intermediate_scores']

    # First analysis: Compare all methods (existing code)
    methods = {
        'Always Tie': intermediate_scores['additional_methods']['always_tie'],
        'Random': intermediate_scores['additional_methods']['random'],
        'Pairwise': intermediate_scores['pairwise_baseline'],
        'Pairwise No Reasoning': intermediate_scores['pairwise_no_reasoning'],
        'Basic (t=0.5)': intermediate_scores['additional_methods']['gpt4o_mini_basic'],
        'Reasoning Before (t=0.5)': intermediate_scores['additional_methods']['gpt4o_mini_reasoning'],
        'Reasoning After (t=0.5)': intermediate_scores['additional_methods']['gpt4o_mini_reasoning_after'],
        '10 Trials (t=0.5)': intermediate_scores['additional_methods']['gpt4o_mini_10trials'],
        'With Reference (t=0.5)': intermediate_scores['additional_methods']['gpt4o_mini_reference']
    }

    # Convert to DataFrame for easier handling
    df = pd.DataFrame(methods)

    # Perform Friedman test
    friedman_stat, p_value = stats.friedmanchisquare(
        *[df[col] for col in df.columns])

    print("Friedman Test Results:")
    print(f"Statistic: {friedman_stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("\nThere are significant differences between methods.")

        # Perform post-hoc Nemenyi test
        post_hoc = posthoc_nemenyi_friedman(df)

        # Create heatmap of p-values
        plt.figure(figsize=(12, 10))
        sns.heatmap(post_hoc, annot=True, cmap='RdYlGn_r', center=0.05)
        plt.title(
            'Post-hoc Nemenyi Test P-values\n(Green indicates significant difference)')
        plt.tight_layout()
        plt.savefig('results/exp4a/statistical_analysis.png')
        plt.close()

        # Print mean ranks
        mean_scores = df.mean()
        ranks = mean_scores.rank(ascending=False)
        print("\nMean Scores and Ranks:")
        for method in mean_scores.index:
            print(
                f"{method:25} Score: {mean_scores[method]:.4f} (Rank: {ranks[method]:.1f})")

        # Print significant differences
        print("\nSignificant Differences (p < 0.05):")
        for i in range(len(post_hoc.columns)):
            for j in range(i+1, len(post_hoc.columns)):
                method1 = post_hoc.columns[i]
                method2 = post_hoc.columns[j]
                p = post_hoc.iloc[i, j]
                if p < 0.05:
                    better_method = method1 if mean_scores[method1] > mean_scores[method2] else method2
                    worse_method = method2 if better_method == method1 else method1
                    print(
                        f"{better_method} is significantly better than {worse_method} (p={p:.4f})")
    else:
        print("\nNo significant differences found between methods.")

    # New analysis: Compare different temperatures for no reasoning
    temps_no_reasoning = {
        f"No Reasoning (t={temp})": scores 
        for temp, scores in intermediate_scores['no_reasoning_temps'].items()
    }
    df_temps_no_reasoning = pd.DataFrame(temps_no_reasoning)
    
    friedman_stat_nr, p_value_nr = stats.friedmanchisquare(
        *[df_temps_no_reasoning[col] for col in df_temps_no_reasoning.columns])

    print("\n\nTemperature Comparison (No Reasoning):")
    print(f"Friedman Test Results:")
    print(f"Statistic: {friedman_stat_nr:.4f}")
    print(f"P-value: {p_value_nr:.4f}")

    if p_value_nr < 0.05:
        post_hoc_nr = posthoc_nemenyi_friedman(df_temps_no_reasoning)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(post_hoc_nr, annot=True, cmap='RdYlGn_r', center=0.05)
        plt.title('Post-hoc Nemenyi Test P-values: No Reasoning Temperatures')
        plt.tight_layout()
        plt.savefig('results/exp4a/statistical_analysis_temps_no_reasoning.png')
        plt.close()

    # Compare different temperatures for reasoning
    temps_reasoning = {
        f"Reasoning (t={temp})": scores 
        for temp, scores in intermediate_scores['with_reasoning_temps'].items()
    }
    df_temps_reasoning = pd.DataFrame(temps_reasoning)
    
    friedman_stat_r, p_value_r = stats.friedmanchisquare(
        *[df_temps_reasoning[col] for col in df_temps_reasoning.columns])

    print("\n\nTemperature Comparison (With Reasoning):")
    print(f"Friedman Test Results:")
    print(f"Statistic: {friedman_stat_r:.4f}")
    print(f"P-value: {p_value_r:.4f}")

    if p_value_r < 0.05:
        post_hoc_r = posthoc_nemenyi_friedman(df_temps_reasoning)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(post_hoc_r, annot=True, cmap='RdYlGn_r', center=0.05)
        plt.title('Post-hoc Nemenyi Test P-values: With Reasoning Temperatures')
        plt.tight_layout()
        plt.savefig('results/exp4a/statistical_analysis_temps_reasoning.png')
        plt.close()


if __name__ == "__main__":
    analyze_experiment_4a()
