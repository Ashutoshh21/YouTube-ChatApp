# YouTube Video Chatbot

A LangChain-based pipeline that fetches YouTube transcripts using the native Python API, chunks the text, creates FAISS embeddings locally, and queries the data.

## Features
- Direct transcript retrieval via `youtube-transcript-api` (no wrappers).
- Robust YouTube video ID parser handling standard, short, and shorts URLs.
- Local vector storage using FAISS and HuggingFace Embeddings (`all-MiniLM-L6-v2`).

![YouTube ChatApp Interface](screenshots/Sample_1.png)


## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd YouTubeChatBot
```

2. Set up your environment (link your existing `.venv` or create a new one):
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Project Structure
- `main.ipynb`: Core notebook containing the manual transcript loader, text splitter, vector store setup, and retriever loop.
- `.gitignore`: Configured to exclude system files, local FAISS indexes, and environments.

