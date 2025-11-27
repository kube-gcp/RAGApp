# MiniRAG With Telegraph API Integratioln

This project implements RAG with Small language model integrated with Telegram API

## Phase 1: Functional working program - Complete ✅

This phase includes:
- Telegraph API integration
- SLM
- Query with 4 Documents embedded in RAG itself

1. Overview

This system provides an AI-powered Telegram bot capable of:

- Accepting user queries

- Retrieving relevant documents using a Mini-RAG (FAISS + embeddings)

- Using an LLM (OpenAI or local) to generate final responses

- Returning answers to Telegram users in real time

The solution uses:

- Python Telegram Bot API

- FAISS as vector store

- SentenceTransformer/OpenAI embedding

Optional LLM for final generation
### Prerequisites

- Python

### Installation

1. Install dependencies:
```bash
    pip install python-telegram-bot
    pip install sentence-transformers
    pip install faiss-cpu
```

2. Register telegram bot. Our bot name is RAGmybot , first name : RAGmybot
and enter Token in line # 6 of telegram-bot.py TOKEN = ""

### Running the Application

**IMPORTANT:** You must start RAG App with python .\telegram-bot.py and watch for Bot is running...

1. **Start RAG Application
```bash
python .\telegram-bot.py
```

Wait until you see:
```
✔  Bot is running...
```

2. **In telegram search for RAGmybot and start with /help or /ask biriyani receipe

## Next Steps

Phase 2 will implement:
- Detailed system design document
- RAG trained with given document 
- Better help documentation

## System Design Archieture
```
+------------------+         +---------------------+
|   Telegram User  | <-----> |   Telegram Bot API  |
+------------------+         +---------------------+
                                     |
                                     v
                          +-----------------------+
                          |  Telegram Bot Server  |
                          |  (Python Application) |
                          +----------+------------+
                                     |
          ----------------------------------------------------
          |                        |                          |
          v                        v                          v
+------------------+    +-----------------------+   +----------------------+
| Mini-RAG Engine  |    |   Vector Store (FAISS)|   | LLM (Local)          |
|  (Retriever)     |    |  stores embeddings    |   | generates response   |
+------------------+    +-----------------------+   +----------------------+
          |
          v
+------------------------------+
|  Document Store (Markdown)   |
+------------------------------+

```

