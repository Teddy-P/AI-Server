from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import AutoTokenizer

# Load the tokenizer and model
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Load the dataset
dataset = load_dataset("json", data_files={"train": "/home/ai_admin/my_ai_project/dataset/chat_dataset.jsonl"}, split="train")


# **NEW CODE: Check the first 5 entries in the dataset**
print("First 5 entries of the dataset:")
print(dataset[:5])  # Print the first 5 entries to verify dataset format



# Tokenize the dataset
def tokenize_function(examples):
    # Tokenize the 'prompt' and 'completion' fields
    inputs = tokenizer(examples["prompt"], truncation=True, padding="max_length", max_length=512)
    outputs = tokenizer(examples["completion"], truncation=True, padding="max_length", max_length=512)

    # Use inputs for the model's input and outputs for the labels
    inputs["labels"] = outputs["input_ids"].copy()
    return inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="/home/ai_admin/my_ai_project/results",          # output directory
    evaluation_strategy="no",        # disable evaluation
    learning_rate=2e-5,              # learning rate
    per_device_train_batch_size=4,   # batch size per device during training
    num_train_epochs=3,              # number of training epochs
    weight_decay=0.01,               # strength of weight decay
    save_steps=10_000,               # save checkpoint every 10,000 steps
    save_total_limit=2,              # limit the number of saved checkpoints
    logging_steps=500,               # Log every 500 steps
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # the model to train
    args=training_args,                  # training arguments
    train_dataset=tokenized_dataset,     # the dataset for training
)

# Start training
trainer.train()

# Save the model and tokenizer
model.save_pretrained("/home/ai_admin/my_ai_project/finetuned_distilgpt2")
tokenizer.save_pretrained("/home/ai_admin/my_ai_project/finetuned_distilgpt2")
