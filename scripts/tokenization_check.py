from transformers import AutoTokenizer

# Load the DistilGPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# Sample text
text = "This is a test sentence for tokenization."

# Tokenize the text
tokens = tokenizer.tokenize(text)

# Convert tokens to IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# Print the token IDs
print("Token IDs:", token_ids)
