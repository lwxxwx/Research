```markdown

# Research

AI Project Testing Codes & Environment Configuration

## Project Overview

This repository contains AI-related testing scripts, environment configuration files, and experimental code. It is designed for quickly reproducing development environments across different devices.

## Directory Structure
.├── test/ # Test scripts directory│
 ├── chromadb_test.py # Chroma DB vector database test│
 ├── download_model.py # Model download script│ 
 ├── env-dependencies-version.py # Environment dependency check│ 
 ├── env_test.py # Environment validation│ 
 ├── faiss_test.py # FAISS vector search test│ 
 ├── frame_test.py # Framework comprehensive test script│ 
 ├── gpu_test.py # GPU availability test│ 
 ├── langchain_all_test.py # LangChain full pipeline test│ 
 ├── langchain_test.py # LangChain basic functions test│ 
 ├── langgraph_test.py # LangGraph workflow test│ 
 ├── pytorch_test.py # PyTorch environment test│ 
 ├── sentence-transformers_test.py # Sentence embedding model test│ 
 └── tf_pt_env.yml # Conda environment file
 ├── .gitignore # Git ignore rules
 └── README.md # Project documentation

## Quick Start

### 1. Clone the Repository
git clone https://github.com/lwxxwx/Research.git
cd Research

### 2. Create and Activate Conda Environment
conda env create -f test/tf_pt_env.yml
conda activate tf_pt_env

### 3. Run Test Scripts
Example 1: Test GPU status
python test/gpu_test.py
Example 2: Run comprehensive framework test
python test/frame_test.py

## Environment Details
Python: 3.11
Main frameworks: PyTorch, TensorFlow, TensorRT, LangChain, LangGraph, ChromaDB, FAISS, Sentence-Transformers
Full dependencies: test/tf_pt_env.yml

### frame_test.py Description
The frame_test.py script is a comprehensive environment validation tool that checks the installation and functionality of key AI frameworks:

    Python & NumPy: Verifies Python and NumPy versions.
    PyTorch & CUDA: Checks PyTorch installation, CUDA availability, GPU details, and performs basic tensor operations.
    TensorFlow: Validates TensorFlow installation, lists available GPUs, and tests tensor operations.
    TensorRT: Confirms TensorRT installation.

The script also includes system-level logging suppression to minimize output noise, making it easier to verify core functionality.
Notes

    This repository is for testing and experimental use only.
    After updating code, use git add . && git commit -m "your message" && git push to sync to GitHub.
    
AI Agent Development Scaffold Documentation
##Project Overview
This is a local AI agent development scaffold built on LangChain + LangGraph + Llama + RAG. It provides a clear directory structure and modular design to facilitate rapid development and extension of AI applications.

##Directory Structure
ai-agent-template/
├── main.py                 # Project entry point
├── .env.example            # Environment variable configuration template
├── .gitignore              # Git ignore file configuration
├── requirements.txt        # Project dependency declarations
├── agent/
│   └── graph.py            # LangGraph workflow definition
├── model/
│   └── llama.py            # Local Llama model loading wrapper
├── rag/
│   └── retriever.py        # RAG knowledge base retrieval tool
├── prompts/                # Prompt template directory
├── memory/                 # Conversation memory module directory
├── tools/                  # Custom tool directory
├── configs/                # Configuration file directory
└── utils/                  # Utility function directory

##Quick Start(Run in VS Code with Conda Environment)
1. Open Project Folder

    Open Visual Studio Code.
    Go to File > Open Folder.
    Select and open your ai-agent-template project directory.
2. Select Python Interpreter

    Click the Python version displayed in the bottom-right corner of VS Code.
    Choose your existing Conda environment from the list.
    Ensure the environment is activated in the terminal.
3. Edit and Run main.py

    Open main.py in VS Code editor.
    Modify test input or logic as needed.
    Click the Run ▶ button in the top-right corner to execute the script.

4. View Results in Terminal
After execution, you can view the running logs and final output in the VS Code terminal.

##Extension Suggestions

    Replace Placeholder Logic: Replace the placeholder functions in model/llama.py and rag/retriever.py with real model loading and retrieval logic.
    Add Prompt Templates: Manage prompt templates for different scenarios in the prompts/ directory.
    Implement Conversation Memory: Add conversation history management in the memory/ directory.
    Integrate Custom Tools: Add custom tools in the tools/ directory to extend the agent's capabilities.
    Add Configuration Management: Manage environment-specific configurations in the configs/ directory.

##Tech Stack

    LangChain: LLM application development framework
    LangGraph: Agent workflow construction
    Llama.cpp: Local Llama model execution
    ChromaDB: Vector database
    Python: Primary development language


```
