from core_4 import evaluate_on_human_preferences_batch, grade_then_compare, pairwise_comparison, pairwise_comparison_no_reasoning
from core import judge_response
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from pathlib import Path
import json

REFERENCE = """[Grading Reference]
1.  Flow: Completely disjointed; sentences and paragraphs lack any logical connection. Wording: Full of grammar, spelling, and syntax errors; sentences are nearly incomprehensible. Creativity: No originality or thought; entirely derivative or nonsensical.
2.  Flow: Very poor; minimal coherence between ideas with abrupt transitions. Wording: Numerous errors that impede understanding; very simplistic or awkward phrasing. Creativity: Little effort to present unique ideas; lacks engagement or depth.
3.  Flow: Some logical connections, but the organization is weak and hard to follow. Wording: Frequent errors; limited vocabulary and repetitive language. Creativity: Lacks originality; relies heavily on clichés or predictable ideas.
4.  Flow: Ideas are somewhat connected, but transitions feel forced or confusing. Wording: Noticeable errors and awkward phrasing; attempts at varied vocabulary fall flat. Creativity: Minimal effort to add originality; occasionally bland or uninspired.
5.  Flow: Adequate structure; some logical progression but with occasional lapses. Wording: Correct but basic; limited variety in sentence structure and vocabulary. Creativity: Meets expectations but does not stand out; lacks memorable elements.
6. Flow: Clear organization with minor hiccups; transitions are functional. Wording: Generally effective with few errors; some variety in language and sentence structure. Creativity: Shows occasional sparks of originality; somewhat engaging.
7. Flow: Smooth and logical progression; ideas are well-connected. Wording: Effective and mostly polished; good vocabulary and sentence variety. Creativity: Demonstrates thoughtful and engaging ideas with some unique touches.
8. Flow: Seamless and cohesive; transitions enhance readability. Wording: Precise and polished with diverse sentence structures and vocabulary. Creativity: Fresh and imaginative ideas; captures the reader's interest.
9. Flow: Flawless and engaging; ideas are intricately woven together. Wording: Highly refined and sophisticated; rich vocabulary used effectively. Creativity: Original and compelling; leaves a lasting impression on the reader.
10. Flow: Perfectly seamless; every sentence and paragraph feels purposeful and natural. Wording: Masterful use of language; evocative and impactful. Creativity: Exceptionally innovative and inspiring; a true standout piece.,
"""


def create_grading_function(temperature: float, reasoning: str = None, num_trials: int = 1, judge: str = "gpt-4o-mini", reference: str = ""):
    """Creates a grading function with fixed temperature and reasoning setting"""
    def grade(question: str, response: str) -> Dict:
        results = judge_response(
            response=response,
            question=question,
            temperature=temperature,
            judge=judge,
            reasoning=reasoning,
            num_trials=num_trials,
            reference=reference
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
    temperatures = np.arange(0, 1.1, 0.2)  # 0.0 to 1.0 in steps of 0.2
    comparison_functions = []

    # Add baseline pairwise comparison (with and without reasoning)
    comparison_functions.append(pairwise_comparison)  # with reasoning
    comparison_functions.append(
        pairwise_comparison_no_reasoning)  # without reasoning

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

    # GPT-4o-mini with reference
    grader = create_grading_function(temperature=0.5, reference=REFERENCE)
    comparison_functions.append(grade_then_compare(grader))

    # Add always-tie baseline
    def always_tie(conversation_a, conversation_b, turn: int) -> str:
        return "tie"
    comparison_functions.append(always_tie)

    # Add random baseline
    def random_choice(conversation_a, conversation_b, turn: int) -> str:
        import random
        return random.choice(["model_a", "model_b", "tie"])
    comparison_functions.append(random_choice)

    # Evaluate all functions on the same set of examples
    results = evaluate_on_human_preferences_batch(
        comparison_functions, cutoff=cutoff)

    accuracies = results['accuracies']
    intermediate_scores = results['intermediate_scores']
    num_examples = results['num_examples']

    # Split results (accuracies)
    baseline_accuracy = accuracies[0]  # Pairwise with reasoning
    # Pairwise without reasoning
    pairwise_no_reasoning_accuracy = accuracies[1]
    # Next 6 results (for temps 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    no_reasoning_accuracies = accuracies[2:8]
    with_reasoning_accuracies = accuracies[8:14]  # Next 6 results
    additional_accuracies = accuracies[14:19]  # Next 5 results
    always_tie_accuracy = accuracies[19]  # Always-tie baseline
    random_accuracy = accuracies[20]  # Random baseline

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
        'Always\ntie',
        'Random\nchoice',
        'Pairwise\nbaseline',
        'Pairwise\nno reasoning',
        'GPT-4o-mini\nbasic',
        'GPT-4o-mini\nw/reasoning',
        'GPT-4o-mini\nw/reasoning after',
        'GPT-4o-mini\n10 trials',
        'GPT-4o-mini\nw/reference'
    ]

    temp_05_accuracies = [
        always_tie_accuracy,  # Always-tie baseline
        random_accuracy,  # Random baseline
        baseline_accuracy,  # Pairwise baseline with reasoning
        pairwise_no_reasoning_accuracy,  # Pairwise baseline without reasoning
        additional_accuracies[0],  # GPT-4o-mini basic
        additional_accuracies[1],  # GPT-4o-mini w/reasoning
        additional_accuracies[2],  # GPT-4o-mini w/reasoning after
        additional_accuracies[3],  # GPT-4o-mini 10 trials
        additional_accuracies[4],  # GPT-4o-mini w/reference
    ]

    plt.bar(methods, temp_05_accuracies)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Alignment Accuracy')
    plt.title('Comparison of Different Methods (Temperature = 0.5)')
    plt.grid(True, axis='y')
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    plt.savefig(exp_dir / 'method_comparison.png')
    plt.close()

    # Save combined results with intermediate scores
    combined_results = {
        "experiment": "4a",
        "description": "Testing temperature effects on human preference alignment",
        "parameters": {
            "cutoff": cutoff,
            "num_examples": num_examples
        },
        "results": {
            "baseline_accuracy": baseline_accuracy,
            "temperatures": temperatures.tolist(),
            "no_reasoning": {
                f"{temp:.1f}": acc for temp, acc in zip(temperatures, no_reasoning_accuracies)
            },
            "with_reasoning": {
                f"{temp:.1f}": acc for temp, acc in zip(temperatures, with_reasoning_accuracies)
            },
            "method_comparison": {
                method: acc for method, acc in zip(methods, temp_05_accuracies)
            }
        },
        "intermediate_scores": {
            "pairwise_baseline": intermediate_scores[0],
            "pairwise_no_reasoning": intermediate_scores[1],
            "no_reasoning_temps": {
                f"temp_{temp:.1f}": scores
                for temp, scores in zip(temperatures, intermediate_scores[2:8])
            },
            "with_reasoning_temps": {
                f"temp_{temp:.1f}": scores
                for temp, scores in zip(temperatures, intermediate_scores[8:14])
            },
            "additional_methods": {
                "gpt4o_mini_basic": intermediate_scores[14],
                "gpt4o_mini_reasoning": intermediate_scores[15],
                "gpt4o_mini_reasoning_after": intermediate_scores[16],
                "gpt4o_mini_10trials": intermediate_scores[17],
                "gpt4o_mini_reference": intermediate_scores[18],
                "always_tie": intermediate_scores[19],
                "random": intermediate_scores[20]
            }
        }
    }

    # Save combined results
    with open(exp_dir / 'combined_results.json', 'w') as f:
        json.dump(combined_results, f, indent=2)

    return combined_results


if __name__ == "__main__":
    results = run_temperature_experiment(cutoff=100)
