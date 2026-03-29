# Insurance Document Q&A Agent

> Built with LangChain, OpenAI, and FAISS — RAG-based AI agent

A conversational AI agent that reads insurance policy documents and answers questions about them in natural language. Built to demonstrate agentic AI development skills using the LangChain framework.

## What It Does

Upload any insurance policy document and ask it questions like:
- "What does this policy cover?"
- "What is excluded from coverage?"
- "How do I make a claim?"

The agent retrieves the relevant sections of the document and uses an LLM to generate accurate, context-aware answers.

## How It Works

This project uses **Retrieval Augmented Generation (RAG)**:

1. The policy document is loaded and split into chunks
2. Each chunk is converted into a vector embedding using OpenAI
3. Chunks are stored in a FAISS vector database
4. When a question is asked, the most relevant chunks are retrieved
5. The LLM (GPT-3.5) uses those chunks to generate an accurate answer

## Tech Stack

| Component | Technology |
|---|---|
| AI Agent Framework | LangChain |
| LLM | OpenAI GPT-3.5-turbo |
| Vector Store | FAISS |
| Embeddings | OpenAI Embeddings |
| Environment Management | python-dotenv |

## Setup
```bash
git clone https://github.com/Tomnair/insurance-doc-agent
cd insurance-doc-agent
python -m venv venv
venv\Scripts\activate
pip install langchain langchain-community langchain-openai langchain-text-splitters faiss-cpu python-dotenv
```

Create a `.env` file with your API key:
```
OPENAI_API_KEY=your-key-here
```

Run the agent:
```bash
python agent.py
```

## Author

**Tom Nair** — Data Scientist & AI Developer  
[LinkedIn](https://linkedin.com/in/tharmindernair) · [GitHub](https://github.com/Tomnair)
```

Press **Ctrl+S**, then in the terminal run:
```
git add .
git commit -m "Add README"
git push
