import openai
import json
from typing import Dict, Any, List
import asyncio
from pathlib import Path
import argparse
import getpass
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")

# Use the newer version of the SDK
client = OpenAI()


def judge_response(
    response: str,
    reasoning: bool = False,
    scale: int = 10,
    temperature: float = 0.5,
    judge: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Judge an LLM response using a specified judge model on a scale from 1-N

    Args:
        response (str): The response from an LLM to evaluate
        reasoning (bool): Whether to ask for reasoning before scoring (defaults to False)
        scale (int): Maximum score value (defaults to 10)
        temperature (float): Temperature for the API call (defaults to 0.5)
        judge (str): The model to use as judge (defaults to "gpt-4o-mini")

    Returns:
        Dict containing the score and explanation
    """

    # Define the function schema based on reasoning parameter
    if reasoning:
        functions = [
            {
                "name": "submit_score",
                "description": "Submit a score and explanation for the LLM response",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "explanation": {
                            "type": "string",
                            "description": "Detailed explanation for the score"
                        },
                        "score": {
                            "type": "integer",
                            "description": f"Score from 1-{scale} ({scale} being best)",
                            "minimum": 1,
                            "maximum": scale
                        },
                    },
                    "required": ["score", "explanation"]
                }
            }
        ]
    else:
        functions = [
            {
                "name": "submit_score",
                "description": "Submit a score for the LLM response",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "score": {
                            "type": "integer",
                            "description": f"Score from 1-{scale} ({scale} being best)",
                            "minimum": 1,
                            "maximum": scale
                        }
                    },
                    "required": ["score"]
                }
            }
        ]

    # Construct the prompt based on reasoning parameter
    if reasoning:
        prompt = f"""You are an expert judge evaluating AI responses. First, provide your detailed reasoning about the response's quality, then give a score from 1-{scale}.

Response to evaluate:
{response}

Please think step by step about the response's strengths and weaknesses before providing your score."""
    else:
        prompt = f"""You are an expert judge evaluating AI responses. Please evaluate the following response on a scale from 1-{scale}.

Response to evaluate:
{response}

Please provide only a score, with no explanation."""

    try:
        completion = client.chat.completions.create(
            model=judge,
            messages=[{"role": "user", "content": prompt}],
            functions=functions,
            function_call={"name": "submit_score"},
            temperature=temperature,
            n=10,
        )

        # Extract the function call
        function_call = completion.choices[0].message.function_call

        # Parse the response
        result = json.loads(function_call.arguments)

        return {
            "score": result["score"],
            "explanation": result.get("explanation", "No explanation requested"),
            "success": True
        }

    except Exception as e:
        return {
            "score": None,
            "explanation": f"Error occurred: {str(e)}",
            "success": False
        }


async def evaluate_all_responses(
    judge_model: str,
    scale: int = 10,
    num_trials: int = 10,
    temperature: float = 0.5
) -> List[float]:
    file_path = Path("data/mt_bench/model_answer/gpt-4.jsonl")
    variances = []
    failed_responses = []
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num > 10:
                break
            print(f"\nProcessing response {line_num}/10 with judge {judge_model}")
            
            try:
                data = json.loads(line)
                response = data['choices'][0]['turns'][0]
                
                # Judge response multiple times
                results = []
                successful_trials = 0
                
                # Run multiple trials
                for trial in range(num_trials):
                    result = judge_response(
                        response,
                        reasoning=False,
                        scale=scale,
                        temperature=temperature,
                        judge=judge_model
                    )
                    
                    if result['success']:
                        results.append(result)
                        successful_trials += 1
                    else:
                        print(f"Trial {trial+1} failed: {result['explanation']}")
                    
                    await asyncio.sleep(1)  # Rate limiting
                
                # Calculate variance if we have any successful results
                scores = [r['score'] for r in results if r['success']]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
                    variances.append(variance)
                    print(f"Variance for response {line_num}: {variance:.4f} ({successful_trials}/{num_trials} trials successful)")
                else:
                    failed_responses.append(line_num)
                    print(f"No successful trials for response {line_num}")
                
            except Exception as e:
                failed_responses.append(line_num)
                print(f"Error processing response {line_num}: {str(e)}")
    
    if not variances:
        raise RuntimeError(f"No successful evaluations completed with judge {judge_model}. Failed responses: {failed_responses}")
    
    return variances


async def main():
    parser = argparse.ArgumentParser(description='Compare judgment variance across models')
    parser.add_argument('--num-trials', type=int, default=10,
                      help='Number of times to judge each response (default: 10)')
    parser.add_argument('--scale', type=int, default=10,
                      help='Maximum score value (default: 10)')
    
    args = parser.parse_args()
    
    judge_models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
    temperatures = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    for temp in temperatures:
        print(f"\n{'='*80}")
        print(f"Starting experiments with temperature {temp}")
        print(f"{'='*80}")
        
        for judge in judge_models:
            try:
                print(f"\nStarting evaluation with judge model: {judge}")
                variances = await evaluate_all_responses(
                    judge,
                    scale=args.scale,
                    num_trials=args.num_trials,
                    temperature=temp
                )
                
                avg_variance = sum(variances) / len(variances)
                print(f"\n{'='*10}")
                print(f"Results for {judge} at temperature {temp}:")
                print(f"Average variance across all responses: {avg_variance:.4f}")
                print(f"Number of responses evaluated: {len(variances)}")
                
                # Include temperature in filename
                output_file = f"results_{judge.replace('-', '_')}_temp_{str(temp).replace('.', '_')}.json"
                results = {
                    "judge_model": judge,
                    "temperature": temp,
                    "individual_variances": variances,
                    "average_variance": avg_variance,
                    "num_responses": len(variances),
                    "parameters": {
                        "scale": args.scale,
                        "num_trials": args.num_trials,
                        "temperature": temp
                    }
                }
                
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {output_file}")
                
            except Exception as e:
                print(f"\nError evaluating with {judge} at temperature {temp}: {str(e)}")
                continue

if __name__ == "__main__":
    asyncio.run(main())
