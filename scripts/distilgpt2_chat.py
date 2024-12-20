import re
import torch
import json
import os
from system_utils import get_current_time, get_current_date, get_system_uptime, get_disk_usage
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
model_name = "/home/ai_admin/my_ai_project/finetuned_distilgpt2"
LOG_DIR = os.path.expanduser("~/my_ai_project/chat_logs")
DATASET_DIR = os.path.expanduser("~/my_ai_project/dataset")
max_tokens = 1024  # Max token limit for DistilGPT-2
greetings = ["hello", "hi", "greetings", "hey", "morning", "afternoon", "evening"]

def save_chat_log(conversation):
    """Saves the conversation to a JSON file in the log directory."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"chat_{timestamp}.json")

    with open(log_file, "w") as file:
        json.dump(conversation, file, indent=4)

    print(f"Chat log saved: {log_file}")

def convert_logs_to_dataset():
    """Converts chat logs into a dataset for fine-tuning."""
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    dataset = []
    for log_file in os.listdir(LOG_DIR):
        if log_file.endswith(".json"):
            log_path = os.path.join(LOG_DIR, log_file)
            with open(log_path, "r") as file:
                try:
                    conversation = json.load(file)
                    for exchange in conversation:
                        user_input = exchange.get("user", "")
                        ai_response = exchange.get("ai", "")
                        correction = exchange.get("correction", "")
                        if user_input and ai_response:
                            dataset.append({
                                "prompt": f"{user_input}\n",
                                "completion": f" {correction or ai_response}\n"
                            })
                except json.JSONDecodeError as e:
                    print(f"Error reading {log_path}: {e}")

    dataset_file = os.path.join(DATASET_DIR, "chat_dataset.jsonl")
    with open(dataset_file, "w") as file:
        for entry in dataset:
            json.dump(entry, file)
            file.write("\n")

    print(f"Dataset created: {dataset_file}")

def main():
    # Model initialization
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("DistilGPT-2 loaded successfully!")
    except Exception as e:
        print(f"Error loading DistilGPT-2: {e}")
        return

    device = torch.device("cpu")
    model = model.to(device)
    print(f"Loaded model: {model_name}")

    # Chat variables
    chat_history_ids = None
    chat_data = []

    print("Welcome to DistilGPT-2 Chat! Type 'exit' to quit.")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        try:
            if any(greeting in user_input.lower() for greeting in greetings):
                response = "Hello! How can I assist you today?"
            elif re.search(r'\btime\b', user_input.lower()):
                response = f"The current time is {get_current_time()}."
            elif re.search(r'\bdate\b', user_input.lower()):
                response = f"Today's date is {get_current_date()}."
            elif re.search(r'\buptime\b', user_input.lower()):
                response = f"The system uptime is {get_system_uptime()}."
            elif re.search(r'\bdisk usage\b', user_input.lower()):
                disk_stats = get_disk_usage()
                response = (
                    f"Disk Usage:\n"
                    f"  Total: {disk_stats['total'] // (1024 ** 3)} GB\n"
                    f"  Used: {disk_stats['used'] // (1024 ** 3)} GB\n"
                    f"  Free: {disk_stats['free'] // (1024 ** 3)} GB\n"
                    f"  Percent Used: {disk_stats['percent_used']:.2f}%"
                )
            else:
                new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt').to(device)
                attention_mask = torch.ones_like(new_user_input_ids).to(device)

                if chat_history_ids is not None:
                    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
                    max_history_length = max_tokens - 150
                    if bot_input_ids.shape[-1] > max_history_length:
                        bot_input_ids = bot_input_ids[:, -max_history_length:]
                        attention_mask = attention_mask[:, -max_history_length:]
                else:
                    bot_input_ids = new_user_input_ids

                chat_history_ids = model.generate(
                    bot_input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=150,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=50
                )

                response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

            print("AI: " + response)

            rating = input("Rate this response (1-5): ")
            while not rating.isdigit() or not (1 <= int(rating) <= 5):
                rating = input("Please enter a valid rating (1-5): ")

            correction = input("Provide a correction (or press Enter to skip): ").strip()

            chat_data.append({
                "user": user_input,
                "ai": response,
                "rating": int(rating),
                "correction": correction
            })

        except Exception as e:
            print("AI: I'm sorry, I couldn't process that. Could you try rephrasing?")
            print(f"Debugging info: {str(e)}")

    if chat_data:
        save_chat_log(chat_data)

if __name__ == "__main__":
    main()