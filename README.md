# RAG-Based Property Graph for Financial Data

A Python-based project that demonstrates how to use Property Graphs for Retrieval Augmented Generation (RAG) in the financial domain. The project combines real-time financial data with graph-based knowledge representation to enhance LLM capabilities.

## About

This project showcases how to:
- Build a property graph from financial data
- Use graph-based context retrieval for RAG
- Leverage graph relationships for better context understanding
- Combine structured (graph) and unstructured (text) data for LLMs

## Features

### 1. Knowledge Graph Construction
- Downloads real-time financial data from Yahoo Finance
- Creates a rich property graph in Neo4j with:
  - Companies, sectors, and industries as nodes
  - Stock prices and financial statements as properties
  - Relationships between entities (BELONGS_TO, OPERATES_IN, etc.)
  - Temporal data relationships

### 2. Graph-Enhanced RAG
- Uses graph traversal for context retrieval
- Leverages relationship information for better context
- Combines multiple data points through graph paths
- Provides structured context for LLM prompts

### 3. Query Capabilities
- Multi-hop relationship queries
- Temporal analysis of financial data
- Similarity-based company comparisons
- Sector and industry analysis

## Requirements

- Python 3.8+
- Neo4j 5.0+
- Required Python packages:
  - yfinance (financial data)
  - pandas & numpy (data processing)
  - neo4j (graph database)
  - python-dotenv (configuration)
  - requests (API calls)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-financial-graph.git
cd rag-financial-graph
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Neo4j:
- Install Neo4j Desktop
- Create a new database
- Set password to 'neo4j123' (or update in code)

## Project Structure

```
rag-financial-graph/
├── knowledge_graph/          # Core package
│   ├── __init__.py          # Package initialization
│   ├── data_downloader.py   # Downloads and processes financial data
│   └── graph_manager.py     # Manages graph operations and RAG context
├── data/                    # Data storage
│   └── financial/          # Financial data files
├── run_financial_graph.py   # Main execution script
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Graph Schema

### Nodes
1. Company
   - Properties: symbol, name, market_cap, employees, description
   - Used for: Entity representation

2. Sector
   - Properties: name
   - Used for: Industry categorization

3. Industry
   - Properties: name
   - Used for: Specific market segmentation

4. StockPrice
   - Properties: date, open, high, low, close, volume
   - Used for: Temporal data points

5. FinancialStatement
   - Properties: date, revenue, net_income, eps, type
   - Used for: Financial metrics

### Relationships
1. BELONGS_TO
   - Company -> Sector
   - Used for: Sector classification

2. OPERATES_IN
   - Company -> Industry
   - Used for: Industry classification

3. HAS_PRICE
   - Company -> StockPrice
   - Used for: Temporal price data

4. HAS_STATEMENT
   - Company -> FinancialStatement
   - Used for: Financial reporting

## RAG Usage Examples

### 1. Company Context Retrieval
```cypher
// Get comprehensive company context
MATCH (c:Company {symbol: 'AAPL'})
OPTIONAL MATCH (c)-[:BELONGS_TO]->(s:Sector)
OPTIONAL MATCH (c)-[:OPERATES_IN]->(i:Industry)
OPTIONAL MATCH (c)-[:HAS_PRICE]->(p:StockPrice)
WHERE p.date >= date() - duration({days: 7})
RETURN c, s, i, collect(p) as recent_prices
```

### 2. Industry Analysis Context
```cypher
// Get industry-wide context
MATCH (s:Sector {name: 'Technology'})<-[:BELONGS_TO]-(c:Company)
OPTIONAL MATCH (c)-[:HAS_STATEMENT]->(f:FinancialStatement)
WHERE f.type = 'INCOME_STATEMENT'
RETURN s.name, collect(distinct c.name) as companies,
       avg(f.revenue) as avg_revenue,
       avg(f.net_income) as avg_net_income
```

### 3. Company Comparison Context
```cypher
// Get comparative context
MATCH (c1:Company {symbol: 'MSFT'})-[:BELONGS_TO]->(s:Sector)<-[:BELONGS_TO]-(c2:Company)
WHERE c1 <> c2
WITH c2, abs(c1.market_cap - c2.market_cap) as market_cap_diff
ORDER BY market_cap_diff
LIMIT 5
RETURN c2.symbol, c2.name, c2.market_cap
```
