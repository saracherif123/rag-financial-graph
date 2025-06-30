# Financial RAG with LangChain, Neo4j, and Gemini

A Retrieval-Augmented Generation (RAG) pipeline for financial data using LangChain, Neo4j, and Gemini (Google Generative AI).

## Quick Start

### 1. Clone and Setup
```bash
git clone git@github.com:saracherif123/rag-financial-graph.git
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
  python graph/data_downloader.py
  ```
- Load company data into Neo4j:
  ```bash
  python graph/data_loader.py
  ```

### 4. Run the Main Script
```bash
python run_financial_graph.py
```
- Ask financial questions interactively in the terminal.

## Example Questions
- What does Microsoft do?
- What is the net income of Microsoft Corporation in 2024?
- Which companies operate in the technology sector?
- What other companies operate in the same sector as Microsoft?

## Notes
- Do **not** commit your `venv/` or `.env` files.
- Data loading to Neo4j is handled by `graph/data_loader.py`.
- All retrieval and LLM logic is handled by LangChain integrations with Gemini and Neo4j.

## Project Structure
```
rag-financial-graph/
├── app/
│   ├── RAG_pipeline.py         # Main RAG pipeline
│   └── streamlit_app.py        # Streamlit web app
├── graph/
│   ├── data_downloader.py
│   └── data_loader.py
    └── convert_json_to_csv.py
├── data/                      # Data files
├── requirements.txt           # Dependencies                      # Your secrets (not tracked)
├── .gitignore                 # Excludes venv, etc.
├── README.md
```
