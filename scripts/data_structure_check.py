import json
from transformers import GPT2Tokenizer

# Paths
MODEL_PATH = "/home/ai_admin/my_ai_project/finetuned_distilgpt2"
DATASET_PATH = "/home/ai_admin/my_ai_project/dataset/chat_dataset.jsonl"

def load_tokenizer(model_path):
    """Load the tokenizer from the fine-tuned model."""
    return GPT2Tokenizer.from_pretrained(model_path)

def load_dataset(dataset_path):
    """Load the dataset, skipping invalid or blank lines."""
    data = []
    with open(dataset_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip blank lines
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping invalid line: {line}")
    return data

def analyze_token_lengths(data, tokenizer):
    """Calculate token length statistics for prompts and completions."""
    input_lengths = [len(tokenizer.tokenize(entry["prompt"])) for entry in data]
    output_lengths = [len(tokenizer.tokenize(entry["completion"])) for entry in data]

    return {
        "max_input_tokens": max(input_lengths, default=0),
        "max_output_tokens": max(output_lengths, default=0),
        "avg_input_tokens": sum(input_lengths) / len(input_lengths) if input_lengths else 0,
        "avg_output_tokens": sum(output_lengths) / len(output_lengths) if output_lengths else 0
    }

def main():
    """Main function to load data, analyze token lengths, and print stats."""
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(MODEL_PATH)

    print("Loading dataset...")
    data = load_dataset(DATASET_PATH)

    if not data:
        print("Dataset is empty or invalid.")
        return

    print("Analyzing token lengths...")
    stats = analyze_token_lengths(data, tokenizer)

    print(f"Max Input Tokens: {stats['max_input_tokens']}")
    print(f"Max Output Tokens: {stats['max_output_tokens']}")
    print(f"Average Input Tokens: {stats['avg_input_tokens']:.2f}")
    print(f"Average Output Tokens: {stats['avg_output_tokens']:.2f}")

if __name__ == "__main__":
    main()
