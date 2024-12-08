from core_4 import evaluate_on_human_preferences_batch, grade_then_compare, pairwise_comparison
from core import judge_response
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from pathlib import Path
import json


def create_grading_function(temperature: float, reasoning: str = None):
    """Creates a grading function with fixed temperature and reasoning setting"""
    def grade(question: str, response: str) -> Dict:
        results = judge_response(
            response=response,
            question=question,
            temperature=temperature,
            judge="gpt-4o-mini",
            reasoning=reasoning,
            num_trials=1
        )
        return results[0]  # Return first (and only) result
    return grade


def run_temperature_experiment(cutoff: int = 100):
    """
    Test different temperature settings and compare with pairwise baseline
    All comparisons use the same randomly sampled responses for fair comparison

    Args:
        cutoff: Number of examples to evaluate for each temperature
    """
    # Create results directory and exp4a subdirectory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    exp_dir = results_dir / "exp4a"
    exp_dir.mkdir(exist_ok=True)

    # Create list of comparison functions for each temperature
    temperatures = np.arange(0, 1.1, 0.1)  # 0.0 to 1.0 in steps of 0.1
    comparison_functions = []

    # Add baseline pairwise comparison as first function
    comparison_functions.append(pairwise_comparison)

    # Add temperature-based comparison functions without reasoning
    for temp in temperatures:
        grader = create_grading_function(temperature=temp, reasoning=None)
        comparison_fn = grade_then_compare(grader)
        comparison_functions.append(comparison_fn)

    # Add temperature-based comparison functions with reasoning="before"
    for temp in temperatures:
        grader = create_grading_function(temperature=temp, reasoning="before")
        comparison_fn = grade_then_compare(grader)
        comparison_functions.append(comparison_fn)

    # Evaluate all functions on the same set of examples
    accuracies = evaluate_on_human_preferences_batch(
        comparison_functions, cutoff=cutoff)

    # Split results
    baseline_accuracy = accuracies[0]
    no_reasoning_accuracies = accuracies[1:12]  # Next 11 results
    with_reasoning_accuracies = accuracies[12:]  # Last 11 results

    # Print results
    print("--------------------------------")
    print(f"Baseline pairwise accuracy: {baseline_accuracy:.3f}")
    print("--------------------------------")
    print("Without reasoning:")
    for temp, acc in zip(temperatures, no_reasoning_accuracies):
        print(f"Temperature {temp:.1f} accuracy: {acc:.3f}")
    print("--------------------------------")
    print("With reasoning:")
    for temp, acc in zip(temperatures, with_reasoning_accuracies):
        print(f"Temperature {temp:.1f} accuracy: {acc:.3f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, no_reasoning_accuracies, 'bo-',
             label='Temperature-based (No reasoning)')
    plt.plot(temperatures, with_reasoning_accuracies, 'go-',
             label='Temperature-based (With reasoning)')
    plt.axhline(y=baseline_accuracy, color='r', linestyle='--',
                label='Pairwise baseline')
    plt.xlabel('Temperature')
    plt.ylabel('Alignment Accuracy')
    plt.title('Temperature vs Human Preference Alignment')
    plt.legend()
    plt.grid(True)
    plt.savefig(exp_dir / 'temperature_alignment.png')
    plt.close()

    # Save combined results
    combined_results = {
        "experiment": "4a",
        "description": "Testing temperature effects on human preference alignment",
        "parameters": {
            "cutoff": cutoff,
        },
        "results": {
            "baseline_accuracy": baseline_accuracy,
            "temperatures": temperatures.tolist(),
            "no_reasoning": {
                temp: acc for temp, acc in zip(temperatures, no_reasoning_accuracies)
            },
            "with_reasoning": {
                temp: acc for temp, acc in zip(temperatures, with_reasoning_accuracies)
            }
        }
    }

    # Save combined results
    with open(exp_dir / 'combined_results.json', 'w') as f:
        json.dump(combined_results, f, indent=2)

    return combined_results


if __name__ == "__main__":
    results = run_temperature_experiment(cutoff=10)
