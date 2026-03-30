# Insurance Document Q&A Assistant

> Built with LangChain, OpenAI, FAISS, and Flask — full-stack RAG AI agent with web UI

A full-stack conversational AI application that reads any insurance policy document and answers questions about it in natural language. Features a web interface, PDF support, and conversation memory.

---

## What It Does

Upload any PDF or text document and have a natural conversation about it:
- "What does this policy cover?"
- "What is excluded from coverage?"
- "How do I make a claim?"
- "What did you just tell me about floods?" — it remembers the conversation

---

## Key Features

Conversational memory — the agent remembers previous questions and answers in the session, enabling natural follow-up questions without repeating context.

PDF and text support — upload any PDF or plain text document and the agent indexes it automatically.

Web interface — built with Flask, runs in the browser with a clean UI, clickable suggestion chips, and a clear conversation button.

RAG architecture — the agent retrieves the most semantically relevant sections of the document before generating each answer, ensuring accuracy and grounding.

Domain-specific — designed around insurance and compliance documents, leveraging actuarial domain knowledge to ask and answer the right questions.

---

## How It Works

The application uses Retrieval Augmented Generation. When you upload a document, it is split into chunks, converted to vector embeddings using OpenAI, and stored in a FAISS vector database. When you ask a question, the most relevant chunks are retrieved and passed to GPT-3.5 along with your conversation history to generate an accurate, grounded answer.

---

## Tech Stack

| Component | Technology |
|---|---|
| Web Framework | Flask |
| AI Agent Framework | LangChain |
| LLM | OpenAI GPT-3.5-turbo |
| Vector Store | FAISS |
| Embeddings | OpenAI Embeddings |
| PDF Processing | PyPDF2 |
| Environment Management | python-dotenv |

---

## Setup
```bash
git clone https://github.com/Tomnair/insurance-doc-agent
cd insurance-doc-agent
python -m venv venv
venv\Scripts\activate
pip install langchain langchain-community langchain-openai langchain-text-splitters faiss-cpu python-dotenv flask pypdf2
```

Create a `.env` file:
```
OPENAI_API_KEY=your-key-here
```

Run the app:
```bash
python agent.py
```

Open your browser and go to `http://127.0.0.1:5000`

---

## Author

**Tom Nair** — Data Scientist & AI Developer
[LinkedIn](https://linkedin.com/in/tharmindernair) · [GitHub](https://github.com/Tomnair)
