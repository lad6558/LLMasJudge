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
from tqdm import tqdm

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
    judge: str = "gpt-4o-mini",
    num_trials: int = 1
) -> List[Dict[str, Any]]:
    """
    Judge an LLM response using a specified judge model on a scale from 1-N

    Args:
        response (str): The response from an LLM to evaluate
        reasoning (bool): Whether to ask for reasoning before scoring (defaults to False)
        scale (int): Maximum score value (defaults to 10)
        temperature (float): Temperature for the API call (defaults to 0.5)
        judge (str): The model to use as judge (defaults to "gpt-4o-mini")
        num_trials (int): Number of trials to run (defaults to 1)

    Returns:
        List of dicts containing the scores and explanations
    """

    # Define the function schema based on reasoning parameter
    if reasoning:
        functions = [
            {
                "name": "submit_score",
                "description": """Submit a score for the LLM response. On a scale of 1 to 10. Break down the scale to sub scores. Novelty from 0-2, instruction following from 0-3, clarity from 0-2, truthiness from 0-2, formatting from 0-1. Don't output the subscores, just the overall score. """,
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
                "description": """Submit a score for the LLM response. On a scale of 1 to 10. 1 means extremely terrible, 5 means mediocre, and 10 means perfect""",
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
            n=num_trials,
        )

        results = []
        # Process all choices from the completion
        for choice in completion.choices:
            try:
                # Extract the function call
                function_call = choice.message.function_call
                # Parse the response
                result = json.loads(function_call.arguments)
                results.append({
                    "score": result["score"],
                    "explanation": result.get("explanation", "No explanation requested"),
                    "success": True
                })
            except json.JSONDecodeError as e:
                results.append({
                    "score": None,
                    "explanation": f"Failed to parse function call response: {str(e)}",
                    "success": False
                })
            except Exception as e:
                results.append({
                    "score": None,
                    "explanation": f"Unexpected error processing choice: {str(e)}",
                    "success": False
                })

        return results

    except openai.RateLimitError as e:
        return [{
            "score": None,
            "explanation": f"Rate limit exceeded: {str(e)}",
            "success": False
        }]
    except openai.APIError as e:
        return [{
            "score": None,
            "explanation": f"OpenAI API error: {str(e)}",
            "success": False
        }]
    except Exception as e:
        return [{
            "score": None,
            "explanation": f"Unexpected error: {str(e)}",
            "success": False
        }]


async def evaluate_all_responses(
    judge_model: str,
    scale: int = 10,
    num_trials: int = 10,
    temperature: float = 0.5,
    cutoff: int = 80
) -> List[float]:
    file_path = Path("data/mt_bench/model_answer/gpt-4.jsonl")
    variances = []
    failed_responses = []
    
    # Count total lines first (up to cutoff)
    total_lines = sum(1 for _ in open(file_path) if _ is not None)
    total_lines = min(cutoff, total_lines)
    
    with open(file_path, 'r') as f:
        # Create progress bar for responses
        pbar = tqdm(total=total_lines, desc=f"Evaluating responses with {judge_model}")
        
        for line_num, line in enumerate(f, 1):
            if line_num > cutoff:
                break
                
            try:
                data = json.loads(line)
                # Validate data structure
                if not isinstance(data, dict) or 'choices' not in data or not data['choices']:
                    raise ValueError("Invalid response data structure")
                if 'turns' not in data['choices'][0] or not data['choices'][0]['turns']:
                    raise ValueError("Invalid turns data structure")
                
                response = data['choices'][0]['turns'][0]
                
                # Judge response multiple times
                results = judge_response(
                    response,
                    reasoning=False,
                    scale=scale,
                    temperature=temperature,
                    judge=judge_model,
                    num_trials=num_trials
                )
                
                successful_trials = sum(1 for result in results if result['success'])
                
                # Calculate variance if we have any successful results
                scores = [result['score'] for result in results if result['success']]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
                    variances.append(variance)
                    pbar.set_postfix({'variance': f"{variance:.4f}", 'trials': f"{successful_trials}/{num_trials}"})
                else:
                    failed_responses.append(line_num)
                    pbar.set_postfix({'status': 'failed'})
                
                
            except json.JSONDecodeError as e:
                failed_responses.append(line_num)
                pbar.set_postfix({'error': 'JSON parse error'})
            except ValueError as e:
                failed_responses.append(line_num)
                pbar.set_postfix({'error': 'Invalid data structure'})
            except Exception as e:
                failed_responses.append(line_num)
                pbar.set_postfix({'error': str(e)[:20]})  # Show first 20 chars of error
            
            pbar.update(1)
        
        pbar.close()
    
    if not variances:
        raise RuntimeError(f"No successful evaluations completed with judge {judge_model}. Failed responses: {failed_responses}")
    
    return variances