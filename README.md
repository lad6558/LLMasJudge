# LLM as Judge

A tool for evaluating LLM responses using another LLM as a judge.

## Usage

```bash
python main.py --model gpt-4 --judge gpt-4o-mini --reasoning
```

### Arguments

- `--model`: Model to evaluate (default: gpt-4)
- `--judge`: Model to use as judge (default: gpt-4o-mini)
- `--num-times`: Number of judgments (default: 10)
- `--scale`: Maximum score (default: 5)
- `--reasoning`: Include reasoning in judgment (default: False)
- `--temperature`: Temperature for API calls (default: 0.5)

## Requirements

- Python 3.7+
- OpenAI API key
