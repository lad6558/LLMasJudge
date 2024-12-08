import asyncio
import json
import argparse
from pathlib import Path
from core import evaluate_all_responses
import asyncio
from tqdm import tqdm

async def run_experiment(num_trials: int = 10, scale: int = 10, cutoff: int = 80):
    """
    Run experiment 3d: Testing model size vs judgment variance
    
    This experiment:
    1. Uses GPT-3.5-turbo, GPT-4o-mini, and GPT-4o to score responses
    2. Tests temperatures from 0.0 to 1.0
    3. Uses detailed scoring scale with reference examples
    4. Adds better error handling and retry logic
    """
    
    # Ensure results directory exists
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Create exp3d subdirectory
    exp_dir = results_dir / "exp3d"
    exp_dir.mkdir(exist_ok=True)
    
    judge_models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
    #temperatures = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    temperatures = [0.3, 0.5, 1.0]
    # Store all results for final analysis
    all_results = {
        "experiment": "3d",
        "description": "Testing model size vs judgment variance with reference examples",
        "parameters": {
            "scale": scale,
            "num_trials": num_trials,
            "cutoff": cutoff
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
        model_pbar = tqdm(judge_models, desc="Testing models", position=1, leave=False)
        
        for judge in model_pbar:
            model_pbar.set_description(f"Model {judge}")
            max_retries = 3  # Add retry logic
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    print(f"\nStarting evaluation with judge model: {judge} (Attempt {retry_count + 1})")
                    variances = await evaluate_all_responses(
                        judge,
                        scale=scale,
                        num_trials=num_trials,
                        temperature=temp,
                        cutoff=cutoff
                    )
                    
                    avg_variance = sum(variances) / len(variances)
                    model_pbar.set_postfix({'avg_variance': f"{avg_variance:.4f}"})
                    
                    # Store results for this temperature and model
                    temp_results[judge] = {
                        "individual_variances": variances,
                        "average_variance": avg_variance,
                        "num_responses": len(variances),
                        "retry_count": retry_count
                    }
                    
                    # Save individual results file
                    model_file = exp_dir / f"{judge.replace('-', '_')}_temp_{str(temp).replace('.', '_')}.json"
                    with open(model_file, 'w') as f:
                        json.dump({
                            "judge_model": judge,
                            "temperature": temp,
                            "individual_variances": variances,
                            "average_variance": avg_variance,
                            "num_responses": len(variances),
                            "retry_count": retry_count,
                            "parameters": {
                                "scale": scale,
                                "num_trials": num_trials,
                                "temperature": temp,
                                "cutoff": cutoff
                            }
                        }, f, indent=2)
                    print(f"Results saved to {model_file}")
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        model_pbar.set_postfix({'error': str(e)[:20]})
                        print(f"\nError evaluating with {judge} at temperature {temp} after {max_retries} attempts: {str(e)}")
                        temp_results[judge] = {
                            "error": str(e),
                            "retry_count": retry_count
                        }
                    else:
                        print(f"\nRetrying... Attempt {retry_count + 1} of {max_retries}")
                        await asyncio.sleep(5)  # Wait before retrying
            
            # Store results for this temperature
            all_results["results"][str(temp)] = temp_results
            model_pbar.close()
        
        temp_pbar.close()
    
    # Save combined results
    combined_file = exp_dir / "combined_results.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {combined_file}")

def main():
    parser = argparse.ArgumentParser(description='Run experiment 3d: Testing model size vs judgment variance')
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
