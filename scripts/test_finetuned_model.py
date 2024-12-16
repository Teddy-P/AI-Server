from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the fine-tuned model and tokenizer
model_name = "/home/ai_admin/my_ai_project/finetuned_distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Function to generate a response from the model
def generate_response(prompt, max_length=50):
    # Tokenize the input and generate attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    attention_mask = inputs["attention_mask"]

    # Generate output using the model
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask
        )

    # Decode the generated output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example prompt
prompt = "How can I improve the performance of my AI model?"
response = generate_response(prompt)
print("Generated Response:", response)
