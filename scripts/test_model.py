from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your fine-tuned model and tokenizer
model_name = "/home/ai_admin/my_ai_project/finetuned_distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate responses
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100, num_return_sequences=1, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example interaction
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = generate_response(user_input)
        print(f"AI: {response}")
