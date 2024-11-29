import openai
import json
from typing import Dict, Any, List
import asyncio
from pathlib import Path
import argparse


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
        completion = openai.ChatCompletion.create(
            model=judge,
            messages=[{"role": "user", "content": prompt}],
            functions=functions,
            function_call={"name": "submit_score"},
            temperature=temperature
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


async def judge_multiple_times(response: str, num_times: int = 10) -> List[Dict[str, Any]]:
    """
    Judge the same response multiple times to study variance

    Args:
        response: The response to judge
        num_times: Number of times to judge the response

    Returns:
        List of judgment results
    """
    results = []
    for i in range(num_times):
        result = judge_response(response)
        results.append(result)
        # Small delay to avoid rate limiting
        await asyncio.sleep(1)
    return results


def print_judgment_results(results: List[Dict[str, Any]], scale: int):
    """Print the judgment results in a readable format"""
    print("\nJudgment Results:")
    print("-" * 80)

    for i, result in enumerate(results, 1):
        print(f"\nTrial {i}:")
        print(f"Explanation: {result['explanation']}")
        print(f"Score: {result['score']}/{scale}")

    # Calculate average score
    scores = [r['score'] for r in results if r['success']]
    if scores:
        avg_score = sum(scores) / len(scores)
        print("\n" + "=" * 80)
        print(f"Average Score: {avg_score:.2f}/{scale}")
        print(
            f"Score Variance: {sum((s - avg_score) ** 2 for s in scores) / len(scores):.2f}")


async def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Judge LLM responses using GPT-4')
    parser.add_argument('--num-times', type=int, default=10,
                        help='Number of times to judge (default: 10)')
    parser.add_argument('--scale', type=int, default=5,
                        help='Maximum score value (default: 5)')
    parser.add_argument('--reasoning', action='store_true',
                        help='Include reasoning in judgment (default: False)')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature for API calls (default: 0.5)')
    parser.add_argument('--model', type=str, default='gpt-4',
                        help='Model name to evaluate (default: gpt-4)')
    parser.add_argument('--judge', type=str, default='gpt-4o-mini',
                        help='Model to use as judge (default: gpt-4o-mini)')

    args = parser.parse_args()

    # Modify file path based on model argument
    file_path = Path(f"data/mt_bench/model_answer/{args.model}.jsonl")

    # Load first response from specified model
    with open(file_path, 'r') as f:
        first_line = f.readline()
        data = json.loads(first_line)
        response = data['choices'][0]['turns'][0]

    print("Response to evaluate:")
    print("-" * 80)
    print(response)

    # Judge it multiple times with specified parameters
    results = []
    for i in range(args.num_times):
        result = judge_response(
            response,
            reasoning=args.reasoning,
            scale=args.scale,
            temperature=args.temperature,
            judge=args.judge
        )
        results.append(result)
        await asyncio.sleep(1)  # Small delay to avoid rate limiting

    # Print results
    print_judgment_results(results, args.scale)

if __name__ == "__main__":
    asyncio.run(main())
