import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from dotenv import load_dotenv
from app.RAG_pipeline import FinancialRAGSystem

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Financial RAG Demo", page_icon="💸")
st.title("💸 Financial Knowledge Graph RAG System")
st.write("Ask questions about companies, their business, financial data, etc.")

# Initialize RAG system (cache to avoid reloading on every interaction)
@st.cache_resource(show_spinner=False)
def get_rag_system():
    return FinancialRAGSystem()

rag = get_rag_system()

question = st.text_input("Enter your financial question:", "")

if st.button("Ask") or question:
    with st.spinner("Searching and generating answer..."):
        result = rag.ask_question(question)
        result_text = result['result']
        if hasattr(result_text, 'content'):
            result_text = result_text.content
        elif isinstance(result_text, dict) and 'content' in result_text:
            result_text = result_text['content']
        st.markdown(f"### 💡 Answer\n{result_text.strip()}")
        if result.get('source_documents'):
            st.markdown("### 📚 Sources")
            for i, doc in enumerate(result['source_documents'], 1):
                lines = doc.page_content.split('\n', 1)
                title = lines[0] if lines else "Source"
                snippet = lines[1][:200] + "..." if len(lines) > 1 and len(lines[1]) > 200 else (lines[1] if len(lines) > 1 else "")
                st.markdown(f"**{i}. {title}**")
                if snippet:
                    st.markdown(f"> {snippet}") 