"""
Run Financial Knowledge Graph with LangChain RAG (Gemini)

This script uses LangChain's Neo4jVector, HuggingFaceEmbeddings, and Gemini (Google Generative AI) for RAG Q&A, configured via environment variables.
"""

import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load from environment
load_dotenv('.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# LangChain Neo4j vector store setup
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Neo4jVector(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    embedding=embeddings,
    index_name="company-embeddings",  # or your actual index name
    node_label="Company",
    text_node_property="description",
    embedding_node_property="embedding"
)

retriever = vectorstore.as_retriever()

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a financial assistant. Use the following context to answer the question.
Context:
{context}

Question: {question}
Answer:
"""
)

if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not set. Set it in your environment to use Gemini.")
else:
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-pro-latest",
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )
    while True:
        user_q = input("\nAsk a financial question (or 'exit'): ")
        if user_q.strip().lower() == 'exit':
            break
        result = qa_chain.invoke({"query": user_q})
        print("\nAnswer:", result["result"])
        print("\nContext used:")
        for doc in result["source_documents"]:
            print("-", doc.page_content[:200], "...") 