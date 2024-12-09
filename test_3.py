import json
from pathlib import Path
import numpy as np
from scipy import stats
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


def load_experiment_results(exp_name: str) -> Dict:
    """Load results from experiment's combined_results.json"""
    results_path = Path(f"results/{exp_name}/combined_results.json")
    if not results_path.exists():
        raise FileNotFoundError(f"No results found for experiment {exp_name}")

    with open(results_path) as f:
        return json.load(f)


def extract_variances(results: Dict, model: str = "gpt-4o-mini") -> Dict[str, List[float]]:
    """Extract variance lists for each temperature for a specific model"""
    variances = {}
    for temp in results["results"]:
        if model in results["results"][temp]:
            model_results = results["results"][temp][model]
            if "individual_variances" in model_results:
                variances[temp] = model_results["individual_variances"]
    return variances


def run_analysis():
    # Load results from each experiment
    experiments = {
        "No Reference": "exp1b",
        "Basic (1,5,10)": "exp3b",
        "Full Scale": "exp3c",
        "Detailed": "exp3d",
        "Subscores": "exp3e"
    }

    # Colors for each experiment type
    colors = {
        "No Reference": "#e74c3c",
        "Basic (1,5,10)": "#3498db",
        "Full Scale": "#2ecc71",
        "Detailed": "#9b59b6",
        "Subscores": "#f1c40f"
    }

    results = {}
    for exp_name, exp_id in experiments.items():
        try:
            results[exp_name] = load_experiment_results(exp_id)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

    # Extract variances for each temperature and experiment
    variances_by_temp = {
        "0.3": {},
        "0.5": {},
        "1.0": {}
    }

    # Collect data for both statistical analysis and plotting
    plot_data = []
    for exp_name, exp_results in results.items():
        exp_variances = extract_variances(exp_results)
        for temp in variances_by_temp:
            if temp in exp_variances:
                variances_by_temp[temp][exp_name] = exp_variances[temp]
                mean_variance = np.mean(exp_variances[temp])
                std_error = np.std(
                    exp_variances[temp]) / np.sqrt(len(exp_variances[temp]))
                plot_data.append({
                    'Temperature': float(temp),
                    'Experiment': exp_name,
                    'Mean Variance': mean_variance,
                    'Std Error': std_error
                })

    # Perform statistical analysis for each temperature
    print("\nStatistical Analysis Results:")
    print("=" * 80)

    for temp in variances_by_temp:
        print(f"\nTemperature {temp}:")
        print("-" * 40)

        # Convert to DataFrame for easier analysis
        data = []
        for exp_name, vars_list in variances_by_temp[temp].items():
            data.extend([(exp_name, v) for v in vars_list])

        df = pd.DataFrame(data, columns=['Experiment', 'Variance'])

        # Calculate mean variance for each experiment
        means = df.groupby('Experiment')['Variance'].mean()
        print("\nMean variances:")
        for exp, mean in means.items():
            print(f"{exp}: {mean:.4f}")

        # Perform Kruskal-Wallis H-test
        experiments_list = list(variances_by_temp[temp].keys())
        if len(experiments_list) > 1:
            h_stat, p_value = stats.kruskal(
                *[variances_by_temp[temp][exp] for exp in experiments_list]
            )
            print(f"\nKruskal-Wallis test:")
            print(f"H-statistic: {h_stat:.4f}")
            print(f"p-value: {p_value:.4f}")

            # If significant difference found, perform pairwise Mann-Whitney U tests
            if p_value < 0.05:
                print("\nPairwise Mann-Whitney U tests:")
                for i, exp1 in enumerate(experiments_list):
                    for exp2 in experiments_list[i+1:]:
                        stat, p = stats.mannwhitneyu(
                            variances_by_temp[temp][exp1],
                            variances_by_temp[temp][exp2],
                            alternative='two-sided'
                        )
                        if p < 0.05:
                            print(f"{exp1} vs {exp2}: p={p:.4f} *")
                        else:
                            print(f"{exp1} vs {exp2}: p={p:.4f}")

        # Create violin plot
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=df, x='Experiment', y='Variance')
        plt.title(f'Variance Distribution by Experiment (Temperature {temp})')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot
        plot_dir = Path("results/exp3e")
        plot_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(plot_dir / f'variance_distribution_temp_{temp}.png')
        plt.close()

    # Create bar plot
    df_plot = pd.DataFrame(plot_data)
    plt.figure(figsize=(15, 8))

    # Set width of bars and positions of the bars
    bar_width = 0.15
    temperatures = sorted(df_plot['Temperature'].unique())
    experiments_list = list(experiments.keys())

    # Create bars
    for i, exp_name in enumerate(experiments_list):
        positions = np.arange(len(temperatures)) + i * bar_width
        exp_data = df_plot[df_plot['Experiment'] == exp_name]

        # Ensure data is sorted by temperature
        exp_data = exp_data.sort_values('Temperature')

        plt.bar(positions,
                exp_data['Mean Variance'],
                bar_width,
                label=exp_name,
                color=colors[exp_name],
                yerr=exp_data['Std Error'],
                capsize=5)

    # Customize the plot
    plt.xlabel('Temperature', fontsize=12)
    plt.ylabel('Mean Variance', fontsize=12)
    plt.title('Comparison of Variance Across Reference Types and Temperatures',
              fontsize=14, pad=20)

    # Set x-ticks in the middle of each group
    group_positions = np.arange(len(temperatures)) + bar_width * 2
    plt.xticks(group_positions, temperatures)

    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save plot
    plt.savefig(plot_dir / 'reference_comparison_bar.png',
                dpi=300,
                bbox_inches='tight')
    plt.close()

    # Print numerical results
    print("\nNumerical Results:")
    print("=" * 80)
    pivot_table = df_plot.pivot_table(
        values='Mean Variance',
        index='Experiment',
        columns='Temperature',
        aggfunc='mean'
    )
    print("\nMean Variances:")
    print(pivot_table.round(4))

    # Calculate percentage improvements over baseline
    print("\nPercentage Improvement Over No Reference:")
    baseline = pivot_table.loc["No Reference"]
    improvements = (baseline - pivot_table) / baseline * 100
    improvements = improvements.drop("No Reference")
    print(improvements.round(2))


if __name__ == "__main__":
    run_analysis()
