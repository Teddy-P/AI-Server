import json

# Path to your dataset
dataset_file = "/home/ai_admin/my_ai_project/dataset/chat_dataset.jsonl"
output_file = "/home/ai_admin/my_ai_project/dataset/edited_chat_dataset.jsonl"

# Function to load and print dataset
def load_and_display_data():
    with open(dataset_file, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    return data

# Function to display the data and allow editing
def edit_data():
    data = load_and_display_data()

    print("Current dataset entries:")
    for i, entry in enumerate(data):
        print(f"Entry {i+1}: {entry['prompt']}")
        print(f"Completion: {entry['completion']}")
        print("-" * 50)

    # Here, you can loop through each entry and allow manual editing.
    for i, entry in enumerate(data):
        print(f"Editing Entry {i+1}:")
        new_prompt = input(f"Edit prompt (current: {entry['prompt']}): ")
        new_completion = input(f"Edit completion (current: {entry['completion']}): ")

        # Only update if the user provides new content
        if new_prompt.strip():
            entry['prompt'] = new_prompt
        if new_completion.strip():
            entry['completion'] = new_completion

    return data

# Function to save the edited dataset
def save_edited_data(edited_data):
    with open(output_file, 'w') as f:
        for entry in edited_data:
            f.write(json.dumps(entry) + "\n")
    print(f"Edited dataset saved to {output_file}")

# Main function to edit and save
def main():
    edited_data = edit_data()
    save_edited_data(edited_data)

if __name__ == "__main__":
    main()
