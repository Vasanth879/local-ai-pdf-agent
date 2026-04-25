import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA

# --- 1. UI CONFIGURATION ---
st.set_page_config(page_title="Local AI PDF Agent", layout="wide")
st.title("📄 Local AI PDF Agent")
st.markdown("---")

# The model name defined in the repository's requirements
MODEL_NAME = "llama3.2:3b"

# --- 2. DATA PROCESSING FUNCTION ---
def process_pdf(file_path):
    """
    This function handles the RAG (Retrieval-Augmented Generation) pipeline:
    1. Loads the PDF
    2. Splits it into chunks
    3. Creates vector embeddings using Ollama
    4. Stores them in a local FAISS database
    """
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Chunking text so the AI can find specific information efficiently
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    final_docs = text_splitter.split_documents(docs)
    
    # Creating embeddings (Local AI logic)
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    vectorstore = FAISS.from_documents(final_docs, embeddings)
    return vectorstore

# --- 3. SIDEBAR: FILE MANAGEMENT ---
with st.sidebar:
    st.header("Document Control")
    uploaded_file = st.file_uploader("Upload your study PDF", type="pdf")
    
    if uploaded_file:
        # Save file locally to allow the loader to read it
        with open("temp_study_doc.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Index Document"):
            with st.spinner("Analyzing and indexing..."):
                st.session_state.vectorstore = process_pdf("temp_study_doc.pdf")
                st.success("PDF ready for querying!")

# --- 4. MAIN CHAT INTERFACE ---
if "vectorstore" in st.session_state:
    user_query = st.text_input("Ask a question about the document:")
    
    if user_query:
        # Initialize the Local LLM via Ollama
        llm = ChatOllama(model=MODEL_NAME)
        
        # Setup the Retrieval Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=st.session_state.vectorstore.as_retriever()
        )
        
        with st.spinner("Searching document..."):
            response = qa_chain.invoke(user_query)
            st.markdown("### AI Response:")
            st.info(response["result"])
else:
    st.warning("Please upload a PDF in the sidebar to begin analysis.")
