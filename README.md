# AI Server Project

## Overview
The AI Server project is designed to create a lightweight and efficient AI system capable of assisting with:
- Basic chat functionality.
- Analyzing and providing insights from documents such as operator manuals, parts books, and electrical prints.
- Server maintenance, including automated scripts and system monitoring.

This project prioritizes open-source solutions and focuses on leveraging Hugging Face's Transformers library.

## Features
- **Lightweight Chat Interface**: A simple and efficient chatbot using DistilGPT-2.
- **Document Analysis**: Support for uploading and analyzing manuals and technical documents.
- **Server Maintenance Assistance**: Tools and scripts for automating server tasks.
- **Web-Based Frontend**: Accessible interface hosted on a separate web server for remote interactions.

## Directory Structure
```
AI-Server/
├── docs/                # Documentation for setup, usage, etc.
├── models/              # Pre-trained and fine-tuned models
├── src/                 # Source code (scripts and utilities)
├── data/                # Datasets and manuals (add to .gitignore if large)
├── tests/               # Test cases
├── .github/             # GitHub workflows for CI/CD
└── README.md
```

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/Teddy-P/AI-Server.git
   cd AI-Server
   ```
2. Set up a Python virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate   # For Linux/macOS
   env\Scripts\activate     # For Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure the environment:
   - Add datasets and documents to the `data/` directory.
   - Download and place pre-trained models in the `models/` directory.

5. Run the server:
   ```bash
   python src/main.py
   ```

## Contributing
We welcome contributions! Here's how you can help:
- Report bugs and suggest features via [Issues](https://github.com/Teddy-P/AI-Server/issues).
- Fork the repository, make changes, and submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

## Future Plans
- Fine-tuning the chatbot model with domain-specific data.
- Enhanced document parsing and querying capabilities.
- Integration with a web-based frontend for user interaction.

---
Feel free to reach out with suggestions or questions!

