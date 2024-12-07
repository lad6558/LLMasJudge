import asyncio
import json
import argparse
from pathlib import Path
from core import evaluate_all_responses
from tqdm import tqdm


async def run_experiment(num_trials: int = 10, scale: int = 10, cutoff: int = 80):
    """
    Run experiment 2a: Testing if step-by-step reasoning reduces judgment variance

    This experiment:
    1. Uses GPT-4o-mini to score GPT-4 responses
    2. Tests temperatures [0.3, 0.5, 1.0]
    3. Requires step-by-step reasoning before scoring
    4. Tests hypothesis that reasoning reduces variance in judgments

    Args:
        num_trials: Number of trials per response
        scale: Score scale (1-N)
        cutoff: Maximum number of responses to evaluate (default: 80)
    """

    # Ensure results directory exists
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Create exp2a subdirectory
    exp_dir = results_dir / "exp2a"
    exp_dir.mkdir(exist_ok=True)

    judge_model = "gpt-4o-mini"
    temperatures = [0.3, 0.5, 1.0]

    # Store all results for final analysis
    all_results = {
        "experiment": "2a",
        "description": "Testing if step-by-step reasoning reduces judgment variance",
        "parameters": {
            "scale": scale,
            "num_trials": num_trials,
            "cutoff": cutoff,
            "judge_model": judge_model
        },
        "results": {}
    }

    # Create progress bar for temperatures
    temp_pbar = tqdm(temperatures, desc="Testing temperatures", position=0)

    for temp in temp_pbar:
        temp_pbar.set_description(f"Temperature {temp:.1f}")
        print(f"\n{'='*80}")
        print(f"Starting experiments with temperature {temp}")
        print(f"{'='*80}")

        try:
            print(f"\nStarting evaluation with reasoning")
            variances = await evaluate_all_responses(
                judge_model,
                scale=scale,
                num_trials=num_trials,
                temperature=temp,
                cutoff=cutoff,
                reasoning="before"
            )

            avg_variance = sum(variances) / len(variances)
            temp_pbar.set_postfix({'avg_variance': f"{avg_variance:.4f}"})

            # Store results for this temperature
            all_results["results"][str(temp)] = {
                "individual_variances": variances,
                "average_variance": avg_variance,
                "num_responses": len(variances)
            }

            # Save individual results file
            temp_file = exp_dir / f"temp_{str(temp).replace('.', '_')}.json"
            with open(temp_file, 'w') as f:
                json.dump({
                    "temperature": temp,
                    "individual_variances": variances,
                    "average_variance": avg_variance,
                    "num_responses": len(variances),
                    "parameters": {
                        "scale": scale,
                        "num_trials": num_trials,
                        "temperature": temp,
                        "cutoff": cutoff,
                        "reasoning": "before"
                    }
                }, f, indent=2)
            print(f"Results saved to {temp_file}")

        except Exception as e:
            print(f"\nError at temperature {temp}: {str(e)}")
            all_results["results"][str(temp)] = {"error": str(e)}

    temp_pbar.close()

    # Save combined results
    combined_file = exp_dir / "combined_results.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {combined_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Run experiment 2a: Testing if reasoning reduces judgment variance')
    parser.add_argument('--num-trials', type=int, default=10,
                        help='Number of times to judge each response (default: 10)')
    parser.add_argument('--scale', type=int, default=10,
                        help='Maximum score value (default: 10)')
    parser.add_argument('--cutoff', type=int, default=80,
                        help='Maximum number of responses to evaluate (default: 10)')

    args = parser.parse_args()

    asyncio.run(run_experiment(
        num_trials=args.num_trials,
        scale=args.scale,
        cutoff=args.cutoff
    ))


if __name__ == "__main__":
    main()
