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
    question: str,
    reasoning: str = None,
    scale: int = 10,
    temperature: float = 0.5,
    judge: str = "gpt-4o-mini",  # gpt-4o-mini is a valid model
    num_trials: int = 1,
    reference: str = ""
) -> List[Dict]:
    """
    Judge a response given its question context

    Args:
        response: The response to evaluate
        question: The question that prompted the response
        reasoning: Whether to include reasoning ("before", "after", or None)
        scale: Rating scale (default 10)
        temperature: Sampling temperature
        judge: Model to use as judge
        num_trials: Number of trials to run

    Returns:
        List of dicts containing:
            - score: int or None if failed
            - explanation: str explanation or error message
            - success: bool indicating if scoring succeeded
            - raw_prompt: str prompt sent to judge
            - raw_response: str raw response from judge
    """

    # Define the function schema based on reasoning parameter
    # "before" and "after" have different orders of arguments, the difference is in the order of score and explanation
    if reasoning == "after":
        functions = [
            {
                "name": "submit_score",
                "description": "Submit a score and explanation for the LLM response",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "score": {
                            "type": "integer",
                            "description": f"Score from 1-{scale} ({scale} being best)",
                            "minimum": 1,
                            "maximum": scale
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Detailed explanation for the score"
                        }
                    },
                    "required": ["score", "explanation"]
                }
            }
        ]
    elif reasoning == "before":
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
                        }
                    },
                    "required": ["explanation", "score"]
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
    if reasoning == "before":
        prompt = f"""[System]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Let's think step by step about the response's quality.

{reference}

[Question]
{question}

[The Start of Assistant's Answer]
{response}
[The End of Assistant's Answer]

Please think step by step about the response's strengths and weaknesses before providing your score on a scale from 1-{scale}."""

    elif reasoning == "after":
        prompt = f"""[System]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below.

{reference}

[Question]
{question}

[The Start of Assistant's Answer]
{response}
[The End of Assistant's Answer]

First provide a score from 1-{scale}, then explain your reasoning step by step."""

    else:
        prompt = f"""[System]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Please evaluate the following response on a scale from 1-{scale}.

{reference}

[Question]
{question}

[The Start of Assistant's Answer]
{response}
[The End of Assistant's Answer]

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
                    "success": True,
                    "raw_prompt": prompt,
                    "raw_response": choice.message.model_dump_json()
                })
            except json.JSONDecodeError as e:
                results.append({
                    "score": None,
                    "explanation": f"Failed to parse function call response: {str(e)}",
                    "success": False,
                    "raw_prompt": prompt,
                    "raw_response": choice.message.model_dump_json() if choice.message else "No response"
                })
            except Exception as e:
                results.append({
                    "score": None,
                    "explanation": f"Unexpected error: {str(e)}",
                    "success": False,
                    "raw_prompt": prompt,
                    "raw_response": choice.message.model_dump_json() if choice.message else "No response"
                })

        return results

    except Exception as e:
        return [{
            "score": None,
            "explanation": f"Unexpected error: {str(e)}",
            "success": False,
            "raw_prompt": prompt,
            "raw_response": "Error: No response received"
        }]


async def evaluate_all_responses(
    judge_model: str,
    scale: int = 10,
    num_trials: int = 10,
    temperature: float = 0.5,
    cutoff: int = 80,
    reasoning: bool = False,
    example_folder: Path = Path("./examples"),
    prefix: str = None
) -> List[float]:
    """Evaluate responses using a judge model

    Args:
        judge_model: Model to use as judge
        scale: Rating scale
        num_trials: Number of trials per response
        temperature: Sampling temperature
        cutoff: Maximum number of responses to evaluate
        reasoning: Whether to include reasoning
        example_folder: Folder to save example judgements
        prefix: Prefix for example judgements file (defaults to {model_name}_{temperature})
    """
    # Set default prefix if none provided
    if prefix is None:
        prefix = f"{judge_model}_{temperature}"

    # Create example folder if it doesn't exist
    example_folder.mkdir(parents=True, exist_ok=True)
    example_judgements = []

    # Load questions
    questions = {}
    with open("data/mt_bench/question.jsonl", "r") as f:
        for line in f:
            q = json.loads(line)
            # Get first turn question
            questions[q["question_id"]] = q["turns"][0]

    file_path = Path("data/mt_bench/model_answer/gpt-4.jsonl")
    variances = []
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

                # Judge response
                results = judge_response(
                    response=response,
                    question=question,
                    reasoning=reasoning,
                    scale=scale,
                    temperature=temperature,
                    judge=judge_model,
                    num_trials=num_trials
                )

                # Save first 5 examples
                if example_judgements is not None and len(example_judgements) < 5:
                    example_judgements.append({
                        # Save prompt from first trial
                        "raw_prompt": results[0]["raw_prompt"],
                        "raw_responses": [r["raw_response"] for r in results]
                    })
                elif example_judgements is not None:
                    example_file = example_folder / \
                        f"{prefix}_example_judgements.json"
                    with open(example_file, "w") as f:
                        json.dump(example_judgements, f, indent=2)
                    example_judgements = None

                successful_trials = sum(
                    1 for result in results if result['success'])

                # Calculate variance if we have any successful results
                scores = [result['score']
                          for result in results if result['success']]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    variance = sum((s - avg_score) **
                                   2 for s in scores) / len(scores)
                    variances.append(variance)
                    pbar.set_postfix(
                        {'variance': f"{variance:.4f}", 'trials': f"{successful_trials}/{num_trials}"})
                else:
                    failed_responses.append(line_num)
                    pbar.set_postfix({'status': 'failed'})

            except json.JSONDecodeError as e:
                failed_responses.append(line_num)
                pbar.set_postfix({'error': 'JSON parse error'})
            except ValueError as e:
                failed_responses.append(line_num)
                pbar.set_postfix({'error': str(e)[:20]})
            except Exception as e:
                failed_responses.append(line_num)
                pbar.set_postfix({'error': str(e)[:20]})

            pbar.update(1)

        pbar.close()

    if not variances:
        raise RuntimeError(
            f"No successful evaluations completed with judge {judge_model}. Failed responses: {failed_responses}")

    return variances
