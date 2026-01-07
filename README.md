# Agentic RAG: AI Knowledge Assistant

A multi-agent RAG (Retrieval-Augmented Generation) system that answers questions about AI, transformers, and attention mechanisms using the "Attention is All You Need" paper.

## Features

- 🤖 **Multi-Agent System**: Router, Retriever, and Answer agents working together
- 🔍 **Hybrid Search**: Automatically routes between PDF search and web search
- 🧠 **Google Gemini LLM**: Uses Gemini 1.5 Flash for fast, accurate responses
- 📚 **HuggingFace Embeddings**: Local embeddings with BAAI/bge-small-en-v1.5
- 💾 **ChromaDB**: Efficient local vector storage
- 🎨 **Gradio Interface**: Clean, interactive web UI

## Setup

### 1. Install Dependencies

```bash
uv sync
```

Or with pip:
```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

**Get API Keys:**
- **Google Gemini API**: https://makersuite.google.com/app/apikey (Free tier available)
- **Tavily Search API**: https://tavily.com/ (Free tier available)

### 3. Run the Application

```bash
python main.py
```

Or with uv:
```bash
uv run python main.py
```

The Gradio interface will launch at `http://localhost:7860`

## How It Works

1. **Router Agent**: Analyzes the question and decides whether to search the PDF or the web
2. **Retriever Agent**: Uses the appropriate tool (PDF search or web search) to find relevant information
3. **Answer Agent**: Synthesizes the retrieved information into a clear, accurate answer

## Architecture

- **LLM**: Google Gemini 1.5 Flash
- **Embeddings**: HuggingFace BAAI/bge-small-en-v1.5
- **Vector Store**: ChromaDB (local)
- **PDF Loader**: PyPDF
- **Web Search**: Tavily
- **Framework**: CrewAI for multi-agent orchestration

## Example Questions

- "What is self-attention?"
- "Explain the transformer architecture"
- "How does multi-head attention work?"
- "What are the latest developments in AI?" (triggers web search)

## Project Structure

```
.
├── main.py              # Main application
├── pyproject.toml       # Dependencies
├── .env                 # API keys (create this)
├── chroma_db/          # Vector database (auto-generated)
└── attention_is_all_you_need.pdf  # Auto-downloaded
```

## Notes

- First run downloads the PDF and creates embeddings (may take 1-2 minutes)
- Subsequent runs are faster as the vector database is cached
- The system automatically determines whether to search the PDF or the web based on your question

## License

MIT

