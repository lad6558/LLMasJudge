import asyncio
import json
import argparse
from pathlib import Path
from core import judge_response
import asyncio
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


async def evaluate_responses(
    judge_model: str,
    temperature: float,
    scale: int,
    num_trials: int,
    cutoff: int,
    exp_dir: Path
) -> list[list[int]]:
    """
    Evaluate responses using a judge model, returning scores for each response.
    Includes reasoning before scoring.
    """
    # Load questions
    questions = {}
    with open("data/mt_bench/question.jsonl", "r") as f:
        for line in f:
            q = json.loads(line)
            questions[q["question_id"]] = q["turns"][0]

    file_path = Path("data/mt_bench/model_answer/gpt-4.jsonl")
    scores_per_response = []
    failed_responses = []

    # Count total lines first (up to cutoff)
    total_lines = sum(1 for _ in open(file_path) if _ is not None)
    total_lines = min(cutoff, total_lines)

    with open(file_path, 'r') as f:
        pbar = tqdm(total=total_lines,
                   desc=f"Evaluating responses with {judge_model}")

        for line_num, line in enumerate(f, 1):
            if line_num > cutoff:
                break

            try:
                data = json.loads(line)
                question_id = data["question_id"]
                question = questions.get(question_id)
                if not question:
                    raise ValueError(f"No question found for ID {question_id}")

                response = data["choices"][0]["turns"][0]

                # Judge response - modified to include reasoning
                results = judge_response(
                    response=response,
                    question=question,
                    reasoning="before",  # Added reasoning parameter
                    scale=scale,
                    temperature=temperature,
                    judge=judge_model,
                    num_trials=num_trials
                )

                # Get scores for this response
                response_scores = [result['score']
                                 for result in results if result['success']]
                
                if response_scores:
                    scores_per_response.append(response_scores)
                    pbar.set_postfix(
                        {'scores': f"{len(response_scores)}/{num_trials}"})
                else:
                    failed_responses.append(line_num)
                    pbar.set_postfix({'status': 'failed'})

            except Exception as e:
                failed_responses.append(line_num)
                pbar.set_postfix({'error': str(e)[:20]})

            pbar.update(1)

        pbar.close()

    if not scores_per_response:
        raise RuntimeError(
            f"No successful evaluations completed with judge {judge_model}. Failed responses: {failed_responses}")

    return scores_per_response


async def run_experiment(num_trials: int = 100, scale: int = 10, cutoff: int = 30):
    """
    Run experiment 1c with reasoning: Testing model size vs scoring distribution
    
    This experiment:
    1. Uses GPT-3.5-turbo, GPT-4o-mini, and GPT-4o to score responses
    2. Tests temperatures 0.3, 0.5, and 1.0
    3. Collects score distributions across multiple responses and trials
    4. Includes reasoning before scoring
    """
    # Ensure results directory exists
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Create exp1c_reasoning subdirectory
    exp_dir = results_dir / "exp1c_reasoning"  # Changed directory name
    exp_dir.mkdir(exist_ok=True)

    judge_models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
    temperatures = [0.3, 0.5, 1.0]

    # Store all results for final analysis
    all_results = {
        "experiment": "1c_reasoning",  # Changed experiment name
        "description": "Testing model size vs scoring distribution with reasoning",  # Updated description
        "parameters": {
            "scale": scale,
            "num_trials": num_trials,
            "cutoff": cutoff,
            "reasoning": "before"  # Added reasoning parameter
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

        temp_results = {}

        # Create progress bar for models
        model_pbar = tqdm(judge_models, desc="Testing models",
                         position=1, leave=False)

        for judge in model_pbar:
            model_pbar.set_description(f"Model {judge}")
            try:
                print(f"\nStarting evaluation with judge model: {judge}")
                all_scores = await evaluate_responses(
                    judge,
                    temperature=temp,
                    scale=scale,
                    num_trials=num_trials,
                    cutoff=cutoff,
                    exp_dir=exp_dir
                )

                # Process scores for each response
                centered_scores = []
                for response_scores in all_scores:
                    median_score = np.median(response_scores)
                    centered = [score - median_score for score in response_scores]
                    centered_scores.extend(centered)

                # Create distribution plot with fixed x-axis limits
                plt.figure(figsize=(10, 6))
                sns.histplot(centered_scores, bins=20, kde=True)
                plt.title(f'Score Distribution for {judge} with Reasoning (temp={temp})')
                plt.xlabel('Centered Scores')
                plt.ylabel('Frequency')
                plt.xlim(-3, 3)  # Fixed x-axis limits
                plt.savefig(exp_dir / f'distribution_reasoning_{judge}_{str(temp).replace(".", "_")}.png')
                plt.close()

                # Store results for this temperature and model
                temp_results[judge] = {
                    "raw_scores": all_scores,
                    "centered_scores": centered_scores,
                    "num_responses": len(all_scores)
                }

                # Save individual results file
                model_file = exp_dir / \
                    f"reasoning_{judge.replace('-', '_')}_temp_{str(temp).replace('.', '_')}.json"
                with open(model_file, 'w') as f:
                    json.dump({
                        "judge_model": judge,
                        "temperature": temp,
                        "raw_scores": all_scores,
                        "centered_scores": centered_scores,
                        "num_responses": len(all_scores),
                        "parameters": {
                            "scale": scale,
                            "num_trials": num_trials,
                            "temperature": temp,
                            "cutoff": cutoff
                        }
                    }, f, indent=2)
                print(f"Results saved to {model_file}")

            except Exception as e:
                model_pbar.set_postfix({'error': str(e)[:20]})
                print(
                    f"\nError evaluating with {judge} at temperature {temp}: {str(e)}")
                temp_results[judge] = {"error": str(e)}

        # Store results for this temperature
        all_results["results"][str(temp)] = temp_results
        model_pbar.close()

    temp_pbar.close()

    # Save combined results
    combined_file = exp_dir / "combined_results_reasoning.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {combined_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Run experiment 1c with reasoning: Testing model size vs scoring distribution')
    parser.add_argument('--num-trials', type=int, default=100,
                       help='Number of times to judge each response (default: 100)')
    parser.add_argument('--scale', type=int, default=10,
                       help='Maximum score value (default: 10)')
    parser.add_argument('--cutoff', type=int, default=30,
                       help='Maximum number of responses to evaluate (default: 30)')

    args = parser.parse_args()

    asyncio.run(run_experiment(
        num_trials=args.num_trials,
        scale=args.scale,
        cutoff=args.cutoff
    ))


if __name__ == "__main__":
    main()
