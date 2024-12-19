import json
import logging
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
MODEL_PATH = "/home/ai_admin/my_ai_project/finetuned_distilgpt2"
DATASET_PATH = "/home/ai_admin/my_ai_project/dataset/chat_dataset.jsonl"

def load_model_and_tokenizer(model_path):
    """Load the tokenizer and model from the specified path."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        logger.info("Model and Tokenizer loaded successfully!")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}")
        raise

def test_model(tokenizer, model):
    """Test the tokenizer and model with a sample input."""
    test_input = "Hello, how are you?"
    tokenized_input = tokenizer(test_input, return_tensors="pt")
    logger.info(f"Tokenized input: {tokenized_input}")

    model_output = model.generate(tokenized_input["input_ids"])
    logger.info(f"Model output: {tokenizer.decode(model_output[0], skip_special_tokens=True)}")

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
                logger.warning(f"Skipping invalid line: {line}")
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

def check_files():
    """Check the existence of model-related files."""
    files = [
        'config.json', 'generation_config.json', 'merges.txt', 'model.safetensors',
        'special_tokens_map.json', 'tokenizer_config.json', 'tokenizer.json', 'vocab.json'
    ]
    for file in files:
        file_path = os.path.join(MODEL_PATH, file)
        if os.path.exists(file_path):
            logger.info(f"{file} exists.")
        else:
            logger.error(f"{file} is missing!")

def main():
    """Main function to load data, analyze token lengths, and print stats."""
    logger.info("Checking model files...")
    check_files()

    logger.info("Loading tokenizer and model...")
    tokenizer, model = load_model_and_tokenizer(MODEL_PATH)

    logger.info("Testing model...")
    test_model(tokenizer, model)

    logger.info("Loading dataset...")
    data = load_dataset(DATASET_PATH)

    if not data:
        logger.error("Dataset is empty or invalid.")
        return

    logger.info("Analyzing token lengths...")
    stats = analyze_token_lengths(data, tokenizer)

    logger.info(f"Max Input Tokens: {stats['max_input_tokens']}")
    logger.info(f"Max Output Tokens: {stats['max_output_tokens']}")
    logger.info(f"Average Input Tokens: {stats['avg_input_tokens']:.2f}")
    logger.info(f"Average Output Tokens: {stats['avg_output_tokens']:.2f}")

if __name__ == "__main__":
    main()
