from datasets import load_dataset
from transformers import AutoTokenizer

# Load your tokenizer
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Load your dataset (replace with your actual dataset location)
dataset = load_dataset("json", data_files={"train": "/home/ai_admin/my_ai_project/dataset/chat_dataset.jsonl"}, split="train")

# Tokenizing the dataset
def tokenize_function(examples):
    return tokenizer(examples['prompt'], truncation=True, padding="max_length", max_length=512)

# Apply tokenization to dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Save tokenized dataset to new file
tokenized_dataset.save_to_disk("/home/ai_admin/my_ai_project/dataset/tokenized_dataset")
