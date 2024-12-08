from core_4 import evaluate_on_human_preferences_batch, grade_then_compare, pairwise_comparison
from core import judge_response
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from pathlib import Path
import json


def create_grading_function(temperature: float, reasoning: str = None, num_trials: int = 1, judge: str = "gpt-4o-mini"):
    """Creates a grading function with fixed temperature and reasoning setting"""
    def grade(question: str, response: str) -> Dict:
        results = judge_response(
            response=response,
            question=question,
            temperature=temperature,
            judge=judge,
            reasoning=reasoning,
            num_trials=num_trials
        )
        return results[0]  # Return first (and only) result
    return grade


def run_temperature_experiment(cutoff: int = 100):
    """Test different temperature settings and compare with pairwise baseline"""
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

    # Add additional comparison functions at temperature 0.5
    # GPT-4o-mini pointwise without reasoning
    grader = create_grading_function(temperature=0.5)
    comparison_functions.append(grade_then_compare(grader))

    # GPT-4o-mini pointwise with reasoning before
    grader = create_grading_function(temperature=0.5, reasoning="before")
    comparison_functions.append(grade_then_compare(grader))

    # GPT-4o-mini pointwise with reasoning after
    grader = create_grading_function(temperature=0.5, reasoning="after")
    comparison_functions.append(grade_then_compare(grader))

    # GPT-4o-mini with 10 trials
    grader = create_grading_function(temperature=0.5, num_trials=10)
    comparison_functions.append(grade_then_compare(grader))

    # GPT-4o-mini with reasoning before and 10 trials
    grader = create_grading_function(
        temperature=0.5, reasoning="before", num_trials=10)
    comparison_functions.append(grade_then_compare(grader))

    # GPT-4o pointwise
    grader = create_grading_function(temperature=0.5, judge="gpt-4o")
    comparison_functions.append(grade_then_compare(grader))

    # GPT-3.5-turbo with reasoning before and 10 trials
    grader = create_grading_function(temperature=0.5, reasoning="before",
                                     num_trials=10, judge="gpt-3.5-turbo")
    comparison_functions.append(grade_then_compare(grader))

    # Evaluate all functions on the same set of examples
    accuracies = evaluate_on_human_preferences_batch(
        comparison_functions, cutoff=cutoff)

    # Split results
    baseline_accuracy = accuracies[0]
    no_reasoning_accuracies = accuracies[1:12]  # Next 11 results
    with_reasoning_accuracies = accuracies[12:23]  # Next 11 results
    additional_accuracies = accuracies[23:]  # Last 7 results

    # Create temperature vs accuracy plot
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

    # Create bar plot for temperature 0.5 comparisons
    plt.figure(figsize=(12, 6))

    # Prepare data for bar plot
    methods = [
        'Pairwise\nbaseline',
        'GPT-4o-mini\nbasic',
        'GPT-4o-mini\nw/reasoning',
        'GPT-4o-mini\nw/reasoning after',
        'GPT-4o-mini\n10 trials',
        'GPT-4o-mini\nw/reasoning\n10 trials',
        'GPT-4o\nbasic',
        'GPT-3.5-turbo\nw/reasoning\n10 trials'
    ]

    temp_05_accuracies = [
        baseline_accuracy,  # Pairwise baseline
        additional_accuracies[0],  # GPT-4o-mini basic
        additional_accuracies[1],  # GPT-4o-mini w/reasoning
        additional_accuracies[2],  # GPT-4o-mini w/reasoning after
        additional_accuracies[3],  # GPT-4o-mini 10 trials
        additional_accuracies[4],  # GPT-4o-mini w/reasoning 10 trials
        additional_accuracies[5],  # GPT-4o basic
        additional_accuracies[6],  # GPT-3.5-turbo w/reasoning 10 trials
    ]

    plt.bar(methods, temp_05_accuracies)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Alignment Accuracy')
    plt.title('Comparison of Different Methods (Temperature = 0.5)')
    plt.grid(True, axis='y')
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    plt.savefig(exp_dir / 'method_comparison.png')
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
            },
            "method_comparison": {
                method: acc for method, acc in zip(methods, temp_05_accuracies)
            }
        }
    }

    # Save combined results
    with open(exp_dir / 'combined_results.json', 'w') as f:
        json.dump(combined_results, f, indent=2)

    return combined_results


if __name__ == "__main__":
    results = run_temperature_experiment(cutoff=10)
