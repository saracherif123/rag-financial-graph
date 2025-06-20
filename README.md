# Financial RAG with LangChain, Neo4j, and Gemini

A minimal Retrieval-Augmented Generation (RAG) pipeline for financial data using LangChain, Neo4j, HuggingFace embeddings, and Gemini (Google Generative AI).

## Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/yourusername/rag-financial-graph.git
cd rag-financial-graph
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the project root with your credentials:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
GOOGLE_API_KEY=your_gemini_api_key
```

### 3. Prepare and Load Data
- Ensure your Neo4j instance is running.
- Download financial data:
  ```bash
  python knowledge_graph/data_downloader.py
  ```
- Load company data into Neo4j:
  ```bash
  python knowledge_graph/data_loader.py
  ```

### 4. Run the Main Script
```bash
python run_financial_graph.py
```
- Ask financial questions interactively in the terminal.

## Example Questions
- What does Microsoft do?
- What is the market capitalization of Amazon?
- Which companies operate in the technology sector?

## Utility
- `test_gemini_key.py`: Check which Gemini models your API key can access.

## Notes
- Do **not** commit your `venv/` or `.env` files.
- Data loading to Neo4j is handled by `knowledge_graph/data_loader.py`.
- All graph and embedding logic is handled by LangChain integrations.

## Project Structure
```
rag-financial-graph/
├── run_financial_graph.py   # Main script
├── requirements.txt        # Dependencies
├── .env                    # Your secrets (not tracked)
├── .gitignore              # Excludes venv, etc.
├── test_gemini_key.py      # Gemini model checker
├── knowledge_graph/        # Data tools and loader
└── data/                   # Data files
```
