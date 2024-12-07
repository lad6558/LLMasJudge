from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv
from enum import Enum

load_dotenv()
client = OpenAI()


class Verdict(str, Enum):
    MODEL_A = "model_a"
    MODEL_B = "model_b"
    TIE = "tie"


def load_dataset_human_preference() -> Path:
    """Path
    Load the MT-bench human judgments dataset and cache it locally.

    Returns:
        Path: Path to the cached dataset JSON file
    """
    output_path = Path("data/mt_bench_human_judgments.json")

    # Skip download if file already exists
    if output_path.exists():
        return output_path

    # Ensure data directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = load_dataset("lmsys/mt_bench_human_judgments")

    # Convert to list of conversation pairs from the 'human' split
    json_data = [
        {
            'question_id': item['question_id'],
            'model_a': item['model_a'],
            'model_b': item['model_b'],
            'winner': item['winner'],
            'judge': item['judge'],
            'conversation_a': item['conversation_a'],
            'conversation_b': item['conversation_b'],
            'turn': item['turn']
        }
        for item in dataset['human']
    ]

    # Save as JSON with indentation
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=4)

    return output_path


def evaluate_on_human_preferences(comparison_function, cutoff: int = None) -> float:
    """
    Evaluate a comparison function against human preferences from MT-bench.

    Args:
        comparison_function: Function that takes conversation_a and conversation_b and returns
                           "model_a", "model_b", or "tie"
        cutoff: Number of examples to evaluate (None for all examples)

    Returns:
        float: Accuracy score where ties count as 0.5 if they match with a model choice
    """
    load_dataset_human_preference()

    with open("data/mt_bench_human_judgments.json", 'r') as f:
        judgments = json.load(f)

    # Randomly sample if cutoff is specified
    if cutoff is not None:
        import random
        judgments = random.sample(judgments, min(cutoff, len(judgments)))

    total_score = 0

    # Create progress bar
    for item in tqdm(judgments, desc="Evaluating responses"):
        # Get model prediction
        pred = comparison_function(
            item['conversation_a'], item['conversation_b'], item['turn'])

        # Get human judgment
        truth = item['winner']

        # Calculate score
        if pred == truth:
            # Direct match
            total_score += 1
        elif "tie" in [pred, truth]:
            total_score += 0.5

    # Calculate accuracy
    accuracy = total_score / len(judgments)

    return accuracy


def pairwise_comparison(conversation_a, conversation_b, turn: int) -> str:
    """
    Compare two conversations and return which one is better.

    Args:
        conversation_a: List of message dictionaries for first conversation
        conversation_b: List of message dictionaries for second conversation
        turn: Which turn in the conversation to evaluate (1-based index)

    Returns:
        str: "model_a", "model_b", or "tie"
    """
    # Calculate the message index for the turn (2 messages per turn: user + assistant)
    msg_idx = (turn - 1) * 2

    # Extract the question and answers from the specific turn
    question = conversation_a[msg_idx]['content']
    answer_a = conversation_a[msg_idx + 1]['content']
    answer_b = conversation_b[msg_idx + 1]['content']

    # Construct the prompt
    prompt = f"""[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two
AI assistants to the user question displayed below. You should choose the assistant that
follows the user's instructions and answers the user's question better. Your evaluation
should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
and level of detail of their responses. Begin your evaluation by comparing the two
responses and provide a short explanation. Avoid any position biases and ensure that the
order in which the responses were presented does not influence your decision. Do not allow
the length of the responses to influence your evaluation. Do not favor certain names of
the assistants. Be as objective as possible.

[User Question]
{question}

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]"""

    try:
        # Define the function schema
        functions = [
            {
                "name": "submit_verdict",
                "description": "Submit a verdict about which response was better",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "explanation": {
                            "type": "string",
                            "description": "Explanation for the verdict"
                        },
                        "verdict": {
                            "type": "string",
                            "enum": [v.value for v in Verdict],
                            "description": "The verdict of which response was better"
                        },
                    },
                    "required": ["explanation", "verdict"]
                }
            }
        ]

        # Get response from OpenAI with function calling
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            functions=functions,
            function_call={"name": "submit_verdict"},
            temperature=0.5,
        )

        # Extract the function call
        function_call = response.choices[0].message.function_call

        # Parse the response
        import json
        result = json.loads(function_call.arguments)

        return result["verdict"]

    except Exception as e:
        print(f"Error in comparison: {str(e)}")
        return Verdict.TIE.value  # Default to tie in case of errors


evaluate_on_human_preferences(
    comparison_function=pairwise_comparison, cutoff=10)
