from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/home/ai_admin/my_ai_project/finetuned_distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

prompt = "Can you explain how to use the ls command in Linux?"
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
outputs = model.generate(
    inputs["input_ids"], 
    attention_mask=inputs["attention_mask"], 
    max_length=50
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Model response: {response}")

