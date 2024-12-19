import torch
import sys
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Paths to the fine-tuned model and tokenizer
MODEL_PATH = "/home/ai_admin/my_ai_project/finetuned_distilgpt2"

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

# Force the model to use the CPU
device = torch.device("cpu")
model.to(device)

def generate_response(prompt, max_length=50):
    """
    Generate a response based on the user's input prompt.
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    attention_mask = inputs['attention_mask'].to(device)
    input_ids = inputs['input_ids'].to(device)

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main(user_input):
    print("AI Server Chat (type 'exit' to quit)")

    if user_input.lower() == "exit":
        print("Goodbye!")
        return

    # Generate response
    response = generate_response(user_input)
    print(f"AI: {response}")

if __name__ == "__main__":
    # Check if an argument was provided (user input passed as a command-line argument)
    if len(sys.argv) > 1:
        user_input = ' '.join(sys.argv[1:])
        main(user_input)
    else:
        print("Error: No input provided.")
