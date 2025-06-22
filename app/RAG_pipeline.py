"""
Financial Knowledge Graph RAG System

A robust Retrieval-Augmented Generation (RAG) pipeline for financial data using:
- LangChain for orchestration
- Neo4j for knowledge graph storage
- HuggingFace embeddings for semantic search
- HuggingFace Inference API for text generation (free tier)
"""

import os
import sys
from typing import List, Dict, Any
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

# Load environment variables
load_dotenv('.env', override=True)

class FinancialRAGSystem:
    def __init__(self):
        """Initialize the Financial RAG System"""
        self.setup_environment()
        self.setup_neo4j_connection()
        self.setup_vectorstore()
        self.setup_llm()
    
    def setup_environment(self):
        """Setup and validate environment variables"""
        self.neo4j_uri = os.getenv('NEO4J_URI')
        self.neo4j_username = os.getenv('NEO4J_USERNAME')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD')
        self.neo4j_database = os.getenv('NEO4J_DATABASE', 'neo4j')
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        
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
    
    def create_vector_index_if_needed(self):
        """Create vector index in Neo4j if it doesn't exist"""
        try:
            with self.neo4j_driver.session(database=self.neo4j_database) as session:
                # Check if index exists
                result = session.run("""
                    SHOW INDEXES 
                    WHERE name = 'company-embeddings'
                """)
                if not result.data():
                    print("Creating vector index...")
                    # Create vector index
                    session.run("""
                        CREATE VECTOR INDEX company-embeddings
                        FOR (c:Company) ON (c.embedding)
                        OPTIONS {indexConfig: {
                            `vector.dimensions`: 384,
                            `vector.similarity_function`: 'cosine'
                        }}
                    """)
                    print("✅ Vector index created")
                else:
                    print("✅ Vector index already exists")
        except Exception as e:
            print(f"Warning: Could not create vector index: {e}")
            print("Continuing without vector index...")
    
    def generate_embeddings_for_companies(self):
        """Generate embeddings for company descriptions and store in Neo4j"""
        try:
            with self.neo4j_driver.session(database=self.neo4j_database) as session:
                # Get companies without embeddings
                result = session.run("""
                    MATCH (c:Company)
                    WHERE c.embedding IS NULL
                    RETURN c.symbol, c.description
                """)
                
                companies = result.data()
                if not companies:
                    print("✅ All companies already have embeddings")
                    return
                
                print(f"Generating embeddings for {len(companies)} companies...")
                
                for company in companies:
                    if company['c.description']:
                        embedding = self.embeddings.embed_query(company['c.description'])
                        session.run("""
                            MATCH (c:Company {symbol: $symbol})
                            SET c.embedding = $embedding
                        """, symbol=company['c.symbol'], embedding=embedding)
                
                print("✅ Embeddings generated and stored")
        except Exception as e:
            print(f"Warning: Could not generate embeddings: {e}")
    
    def setup_vectorstore(self):
        """Setup Neo4j vector store"""
        try:
            # Create index and generate embeddings first
            self.create_vector_index_if_needed()
            self.retriever = BasicRetriever(neo4j_driver=self.neo4j_driver, database=self.neo4j_database)
            print("✅ Basic keyword retriever setup complete")
            
        except Exception as e:
            print(f"Warning: Vector store setup failed: {e}")
            print("Falling back to basic retrieval...")
            self.retriever = BasicRetriever(neo4j_driver=self.neo4j_driver, database=self.neo4j_database)
    
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
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question and get response with context"""
        print(f"🔍 Searching for: {question}")
        
        # Get relevant documents using our working retrieval system
        try:
            docs = self.retriever.get_relevant_documents(question)
            print(f"📚 Found {len(docs)} relevant documents")
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return {
                "result": f"Error during retrieval: {e}",
                "source_documents": []
            }
        
        if not self.llm:
            # No LLM available - return documents directly
            if docs:
                summary = "Based on the available financial data:\n\n"
                for i, doc in enumerate(docs, 1):
                    content = doc.page_content
                    if "Company:" in content:
                        company_info = content.split("\n")[0]
                        description = content.split("\n", 1)[1] if "\n" in content else ""
                        summary += f"{i}. {company_info}\n{description[:300]}...\n\n"
                return {
                    "result": summary,
                    "source_documents": docs
                }
            else:
                return {
                    "result": "No relevant information found. Try asking about specific companies like Apple, Microsoft, Amazon, etc.",
                    "source_documents": []
                }
        
        try:
            # Use LLM to generate response
            if docs:
                # Create context from documents
                context = "\n\n".join([doc.page_content for doc in docs])
                prompt = f"""You are a financial assistant. Answer the question based on the provided context.

Context:
{context}

Question: {question}

Answer:"""
                
                # Call LLM
                response = self.llm.invoke(prompt)
                return {
                    "result": response,
                    "source_documents": docs
                }
            else:
                return {
                    "result": "No relevant information found in the knowledge graph.",
                    "source_documents": []
                }
        except Exception as e:
            print(f"⚠️  LLM error: {e}")
            # Fallback to simple summary
            if docs:
                summary = "Based on the available financial data:\n\n"
                for i, doc in enumerate(docs, 1):
                    content = doc.page_content
                    if "Company:" in content:
                        company_info = content.split("\n")[0]
                        description = content.split("\n", 1)[1] if "\n" in content else ""
                        summary += f"{i}. {company_info}\n{description[:300]}...\n\n"
                return {
                    "result": summary,
                    "source_documents": docs
                }
            else:
                return {
                    "result": "No relevant information found. Try asking about specific companies like Apple, Microsoft, Amazon, etc.",
                    "source_documents": []
                }
    
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

class BasicRetriever(BaseRetriever):
    """Custom retriever that uses keyword-based search"""
    
    neo4j_driver: any = Field(exclude=True)
    database: str
    stopwords: set = Field(default_factory=lambda: {
        "what", "does", "do", "the", "and", "or", "is", "a", "an", "of", "in", "on", "for", "to", "with", "by", "at", "as", "from", "that", "this", "it"
    }, exclude=True)
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant to the query"""
        clean_query = re.sub(r'[^\w\s]', '', query.lower())
        keywords = [w for w in clean_query.split() if len(w) > 2 and w not in self.stopwords]
        print(f"DEBUG: Extracted keywords: {keywords}")
        with self.neo4j_driver.session(database=self.database) as session:
            docs = []
            seen = set()
            for word in keywords:
                print(f"DEBUG: Searching for keyword: {word}")
                result = session.run("""
                    MATCH (c:Company)
                    WHERE toLower(c.description) CONTAINS $word
                       OR toLower(c.name) CONTAINS $word
                       OR toLower(c.symbol) CONTAINS $word
                    RETURN c.symbol, c.name, c.description
                    LIMIT 1
                """, word=word)
                for record in result:
                    key = record['c.symbol']
                    if key not in seen:
                        seen.add(key)
                        content = f"Company: {record['c.name']} ({record['c.symbol']})\n{record['c.description']}"
                        docs.append(Document(page_content=content))
            return docs
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version - just call the sync version for now"""
        return self._get_relevant_documents(query)

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