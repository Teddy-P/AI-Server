import subprocess
import time

def run_chat_model(input_text):
    try:
        # Run the chat_model.py script with the input text as a command-line argument
        process = subprocess.Popen(
            ['python3', 'my_ai_project/scripts/chat_model.py', input_text],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = process.communicate()

        # Check for errors
        if stderr:
            print(f"Error: {stderr}")
        return stdout
    except Exception as e:
        print(f"Error during execution: {e}")
        return None

def test_chat_model():
    test_inputs = [
        "Hi.",
        "Can you tell me the time?",
        "What is the capital of France?",
        "Tell me a joke.",
        "exit"
    ]

    for input_text in test_inputs:
        print(f"Testing input: {input_text}")
        response = run_chat_model(input_text)
        print(f"Response: {response}")

        # You can add logic here to suggest improvements
        if "Hi." in response:
            print("Suggestion: The model is not engaging well; consider improving the conversational flow.")
        if "Can you tell me the time?" in response:
            print("Suggestion: The model should respond with the current time or an appropriate message.")
        if "exit" in response:
            print("Test complete. Exiting.")
            break
        time.sleep(1)  # Pause for readability between tests

if __name__ == "__main__":
    test_chat_model()
