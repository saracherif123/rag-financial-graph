"""
Financial Knowledge Graph RAG System

A robust Retrieval-Augmented Generation (RAG) pipeline for financial data using:
- LangChain for orchestration
- Neo4j for knowledge graph storage
- HuggingFace embeddings for semantic search
- HuggingFace Inference API for text generation (free tier)
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_neo4j import Neo4jVector
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
import re
from pydantic import Field, SkipValidation
from graph.data_loader import (
    load_companies_to_neo4j,
    load_json_statements_to_neo4j,
    load_stock_prices_to_neo4j,
    COMPANY_FILE,
    BALANCE_SHEET_FILE,
    INCOME_STATEMENT_FILE,
    CASH_FLOW_FILE,
    STOCK_PRICES_FILE,
    CREATE_BALANCE_SHEET_QUERY,
    CREATE_INCOME_STATEMENT_QUERY,
    CREATE_CASH_FLOW_QUERY
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv('.env', override=True)

class FinancialRAGSystem:
    def __init__(self):
        """Initialize the Financial RAG System"""
        self.setup_environment()
        self.setup_neo4j_connection()
        self.load_all_data_to_neo4j()
        self.setup_faiss_vectorstore()
        self.setup_llm()
        self.load_company_name_symbol_mapping()
    
    def setup_environment(self):
        """Setup and validate environment variables"""
        self.neo4j_uri = os.getenv('NEO4J_URI')
        self.neo4j_username = os.getenv('NEO4J_USERNAME')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD')
        self.neo4j_database = os.getenv('NEO4J_DATABASE', 'neo4j')
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        # Use a well-supported model for embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        if not all([self.neo4j_uri, self.neo4j_username, self.neo4j_password]):
            raise ValueError("Missing Neo4j credentials in .env file")
        if not self.google_api_key:
            print("Warning: GOOGLE_API_KEY not set. RAG will work but without LLM generation.")
    
    def setup_neo4j_connection(self):
        """Setup Neo4j connection and verify connectivity"""
        try:
            self.neo4j_driver = GraphDatabase.driver(
                self.neo4j_uri, 
                auth=(self.neo4j_username, self.neo4j_password)
            )
            # Test connection
            with self.neo4j_driver.session(database=self.neo4j_database) as session:
                result = session.run("RETURN 1 as test")
                result.single()
            print("✅ Neo4j connection established")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j: {e}")
    
    def setup_faiss_vectorstore(self):
        """Extract all data from Neo4j, create Documents, and build a FAISS index for semantic search."""
        try:
            docs = []
            # Companies
            with self.neo4j_driver.session(database=self.neo4j_database) as session:
                result = session.run("MATCH (c:Company) RETURN c.symbol AS symbol, c.name AS name, c.description AS description")
                for record in result:
                    content = f"Company: {record['name']} ({record['symbol']})\n{record['description']}"
                    docs.append(Document(page_content=content, metadata={"type": "Company", "symbol": record['symbol']}))
                # Balance Sheets
                result = session.run("MATCH (b:BalanceSheet) RETURN b.symbol AS symbol, b.date AS date, b AS props")
                for record in result:
                    props = record['props']
                    summary = f"BalanceSheet for {props.get('symbol', '')} on {props.get('date', '')}: " + ", ".join([f"{k}: {v}" for k, v in props.items() if k not in ['symbol', 'date', 'embedding']])
                    docs.append(Document(page_content=summary, metadata={"type": "BalanceSheet", "symbol": props.get('symbol', ''), "date": props.get('date', '')}))
                # Income Statements
                result = session.run("MATCH (i:IncomeStatement) RETURN i.symbol AS symbol, i.date AS date, i AS props")
                for record in result:
                    props = record['props']
                    summary = f"IncomeStatement for {props.get('symbol', '')} on {props.get('date', '')}: " + ", ".join([f"{k}: {v}" for k, v in props.items() if k not in ['symbol', 'date', 'embedding']])
                    docs.append(Document(page_content=summary, metadata={"type": "IncomeStatement", "symbol": props.get('symbol', ''), "date": props.get('date', '')}))
                # Cash Flows
                result = session.run("MATCH (f:CashFlow) RETURN f.symbol AS symbol, f.date AS date, f AS props")
                for record in result:
                    props = record['props']
                    summary = f"CashFlow for {props.get('symbol', '')} on {props.get('date', '')}: " + ", ".join([f"{k}: {v}" for k, v in props.items() if k not in ['symbol', 'date', 'embedding']])
                    docs.append(Document(page_content=summary, metadata={"type": "CashFlow", "symbol": props.get('symbol', ''), "date": props.get('date', '')}))
                # Stock Prices
                result = session.run("MATCH (s:StockPrice) RETURN s.Symbol AS symbol, s.Date AS date, s AS props")
                for record in result:
                    props = record['props']
                    summary = f"StockPrice for {props.get('Symbol', '')} on {props.get('Date', '')}: " + ", ".join([f"{k}: {v}" for k, v in props.items() if k not in ['Symbol', 'Date', 'embedding']])
                    docs.append(Document(page_content=summary, metadata={"type": "StockPrice", "symbol": props.get('Symbol', ''), "date": props.get('Date', '')}))
            self.faiss_index = FAISS.from_documents(docs, self.embeddings)
            print(f"✅ FAISS vector store setup complete with {len(docs)} documents.")
        except ImportError:
            print("❌ faiss-cpu is not installed. Please run 'pip install faiss-cpu' to enable semantic search.")
            self.faiss_index = None
        except Exception as e:
            print(f"❌ FAISS vector store setup failed: {e}")
            self.faiss_index = None
    
    def setup_llm(self):
        if self.google_api_key:
            try:
                print("Setting up Gemini LLM (models/gemini-1.5-pro-latest)...")
                self.llm = ChatGoogleGenerativeAI(
                    model="models/gemini-1.5-pro-latest",
                    google_api_key=self.google_api_key
                )
                print("✅ Gemini LLM setup complete")
            except Exception as e:
                print(f"⚠️  Gemini LLM setup failed: {e}")
                print("Continuing with RAG retrieval only...")
                self.llm = None
        else:
            self.llm = None
            print("⚠️  No Gemini LLM available (missing GOOGLE_API_KEY)")
    
    def load_company_name_symbol_mapping(self):
        """Load company name to symbol mapping from company_info.json"""
        self.company_name_to_symbol = {}
        self.symbol_to_company_name = {}
        company_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'financial', 'company_info.json')
        if os.path.exists(company_file):
            with open(company_file, 'r') as f:
                companies = json.load(f)
            for company in companies:
                name = company.get('name', '').lower()
                symbol = company.get('symbol', '').upper()
                if name and symbol:
                    self.company_name_to_symbol[name] = symbol
                    self.symbol_to_company_name[symbol] = name

    def get_symbol_from_question(self, question: str) -> str:
        """Try to find a company name in the question and return its symbol if found."""
        question_lower = question.lower()
        for name, symbol in self.company_name_to_symbol.items():
            if name in question_lower:
                return symbol
        return None

    def replace_company_name_with_symbol(self, question: str) -> str:
        """Replace company name in the question with its symbol if found."""
        for name, symbol in self.company_name_to_symbol.items():
            pattern = re.compile(re.escape(name), re.IGNORECASE)
            if pattern.search(question):
                question = pattern.sub(symbol, question)
        return question

    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question and get response with context using FAISS semantic search. Preprocess to map company names to symbols."""
        # Preprocess question to replace company names with symbols
        processed_question = self.replace_company_name_with_symbol(question)
        print(f"🔍 Searching for: {processed_question}")
        if not self.faiss_index:
            print("❌ FAISS index is not available. Returning no results.")
            return {"result": "Semantic search is not available. Please check your FAISS setup.", "source_documents": []}
        try:
            docs = self.faiss_index.similarity_search(processed_question, k=5)
            print(f"📚 Found {len(docs)} relevant documents")
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return {"result": f"Error during retrieval: {e}", "source_documents": []}
        if not self.llm:
            # No LLM available - return documents directly
            if docs:
                summary = "Based on the available financial data:\n\n"
                for i, doc in enumerate(docs, 1):
                    content = doc.page_content
                    summary += f"{i}. {content[:300]}...\n\n"
                return {"result": summary, "source_documents": docs}
            else:
                return {"result": "No relevant information found.", "source_documents": []}
        try:
            if docs:
                context = "\n\n".join([doc.page_content for doc in docs])
                prompt = f"""You are a financial assistant. Answer the question based on the provided context.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"""
                response = self.llm.invoke(prompt)
                return {"result": response, "source_documents": docs}
            else:
                return {"result": "No relevant information found in the knowledge base.", "source_documents": []}
        except Exception as e:
            print(f"⚠️  LLM error: {e}")
            if docs:
                summary = "Based on the available financial data:\n\n"
                for i, doc in enumerate(docs, 1):
                    content = doc.page_content
                    summary += f"{i}. {content[:300]}...\n\n"
                return {"result": summary, "source_documents": docs}
            else:
                return {"result": "No relevant information found.", "source_documents": []}
    
    def run_interactive(self):
        """Run interactive Q&A session"""
        print("\n" + "="*60)
        print("🤖 Financial Knowledge Graph RAG System")
        print("="*60)
        print("Ask questions about companies, their business, financial data, etc.")
        print("Type 'exit' to quit, 'help' for example questions")
        print("="*60)
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() == 'exit':
                    print("👋 Goodbye!")
                    break
                
                if question.lower() == 'help':
                    self.show_help()
                    continue
                
                if not question:
                    continue
                
                print("\n🔍 Searching...")
                result = self.ask_question(question)
                
                print("\n" + "="*60)
                print("💡 Answer:\n")
                # Extract answer text cleanly
                result_text = result['result']
                if hasattr(result_text, 'content'):
                    result_text = result_text.content
                elif isinstance(result_text, dict) and 'content' in result_text:
                    result_text = result_text['content']
                print(result_text.strip())
                print("="*60)
                
                if result.get('source_documents'):
                    print("\n📚 Sources:")
                    for i, doc in enumerate(result['source_documents'], 1):
                        lines = doc.page_content.split('\n', 1)
                        title = lines[0] if lines else "Source"
                        snippet = lines[1][:200] + "..." if len(lines) > 1 and len(lines[1]) > 200 else (lines[1] if len(lines) > 1 else "")
                        print(f"{i}. {title}")
                        if snippet:
                            print(f"   {snippet}")
                print("="*60)
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def show_help(self):
        """Show example questions"""
        examples = [
            "What does Apple Inc. do?",
            "What is Microsoft's business model?",
            "Which companies are in the technology sector?",
            "What is the market capitalization of Amazon?",
            "Tell me about NVIDIA's business",
            "Which companies focus on software?",
            "What does Tesla do?",
            "Tell me about companies in the semiconductor industry"
        ]
        
        print("\n💡 Example questions:")
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example}")
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'neo4j_driver'):
            self.neo4j_driver.close()
    
    def load_all_data_to_neo4j(self):
        """Load all available data into Neo4j from the data/financial directory"""
        try:
            # Load companies
            if os.path.exists(COMPANY_FILE):
                with open(COMPANY_FILE, 'r') as f:
                    companies = json.load(f)
                load_companies_to_neo4j(self.neo4j_driver, companies)
            # Load balance sheets
            load_json_statements_to_neo4j(self.neo4j_driver, BALANCE_SHEET_FILE, CREATE_BALANCE_SHEET_QUERY, 'balance sheet')
            # Load income statements
            load_json_statements_to_neo4j(self.neo4j_driver, INCOME_STATEMENT_FILE, CREATE_INCOME_STATEMENT_QUERY, 'income statement')
            # Load cash flows
            load_json_statements_to_neo4j(self.neo4j_driver, CASH_FLOW_FILE, CREATE_CASH_FLOW_QUERY, 'cash flow')
            # Load stock prices
            load_stock_prices_to_neo4j(self.neo4j_driver, STOCK_PRICES_FILE)
            print("✅ All available data loaded into Neo4j.")
        except Exception as e:
            print(f"⚠️  Data loading error: {e}")

def main():
    """Main function to run the Financial RAG System"""
    try:
        rag_system = FinancialRAGSystem()
        rag_system.run_interactive()
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        sys.exit(1)
    finally:
        if 'rag_system' in locals():
            rag_system.close()

if __name__ == "__main__":
    main() 