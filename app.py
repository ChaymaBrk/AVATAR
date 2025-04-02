import os
import json
import shutil
import numpy as np
import asyncio
from typing import List, Union
from pathlib import Path
import streamlit as st
import tempfile
import time

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from PyPDF2 import PdfReader

from dotenv import load_dotenv
import logging
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Azure OpenAI Configuration
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Working directory for storing embeddings and LightRAG instance
BASE_WORKING_DIR = "./dickens"  # Base directory for all RAG instances
STORAGE_DIR = "./rag_storage"
PDF_DIR = "./legal_documents"  # Directory for PDF files

# Ensure storage directories exist
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(BASE_WORKING_DIR, exist_ok=True)

# Initialize session state
if 'rag' not in st.session_state:
    st.session_state.rag = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'working_dirs' not in st.session_state:
    st.session_state.working_dirs = []

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""

def process_documents(document_paths: List[Union[str, Path]]) -> List[str]:
    """Process multiple documents, handling both PDF and text files."""
    documents = []
    for doc_path in document_paths:
        doc_path = str(doc_path)
        try:
            if doc_path.lower().endswith('.pdf'):
                text = extract_text_from_pdf(doc_path)
                if text:
                    documents.append(text)
            else:
                with open(doc_path, 'r', encoding='utf-8') as file:
                    documents.append(file.read())
        except Exception as e:
            logging.error(f"Error processing document {doc_path}: {e}")
            continue
    return documents

def llm_model_func(
    prompt, 
    system_prompt=None, 
    history_messages=None, 
    keyword_extraction=False, 
    **kwargs
) -> str:
    """Azure OpenAI chat completion function."""
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    chat_completion = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        temperature=kwargs.get("temperature", 0),
        top_p=kwargs.get("top_p", 1),
        n=kwargs.get("n", 1),
    )
    return chat_completion.choices[0].message.content

def embedding_func(texts: list[str]) -> np.ndarray:
    """Azure OpenAI embedding function."""
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_EMBEDDING_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
    
    embedding = client.embeddings.create(
        model=AZURE_EMBEDDING_DEPLOYMENT, 
        input=texts
    )
    
    embeddings = [item.embedding for item in embedding.data]
    return np.array(embeddings)

async def async_insert_documents(rag, documents):
    """Asynchronous document insertion with error handling."""
    try:
        await rag.ainsert(documents)
    except KeyError as e:
        logging.warning(f"Encountered KeyError during document insertion: {e}")
        pass
    except Exception as e:
        logging.error(f"Unexpected error during document insertion: {e}")
        raise

def save_rag_instance(rag, storage_path):
    """
    Comprehensively save LightRAG instance and all related files.
    
    This function will:
    1. Save configuration details
    2. Copy all relevant storage files to the storage directory
    """
    # Ensure storage path exists
    os.makedirs(storage_path, exist_ok=True)
    
    # Save RAG configuration
    config = {
        'working_dir': rag.working_dir,
        'embedding_func_config': {
            'embedding_dim': rag.embedding_func.embedding_dim,
            'max_token_size': rag.embedding_func.max_token_size
        }
    }
    
    # Save configuration file
    with open(os.path.join(storage_path, 'rag_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # List of expected LightRAG storage files and their patterns
    storage_files = [
        'kv_store_doc_status',
        'kv_store_full_docs',
        'kv_store_llm_response_cache',
        'kv_store_text_chunks',
        'vdb_chunks_',
        'vdb_entities',
        'vdb_relationships'
    ]
    
    # Copy relevant files from working directory to storage directory
    for filename in os.listdir(rag.working_dir):
        if any(sf in filename for sf in storage_files):
            src_path = os.path.join(rag.working_dir, filename)
            dst_path = os.path.join(storage_path, filename)
            
            # Copy the file
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
                logging.info(f"Copied {filename} to storage directory")
            elif os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
                logging.info(f"Copied directory {filename} to storage directory")
    
    logging.info(f"RAG instance and related files saved to {storage_path}")

def process_uploaded_files(uploaded_files):
    """Process uploaded PDF files and save them to the legal_documents directory."""
    processed_files = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            # Create a temporary file to save the uploaded PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Move the temporary file to the legal_documents directory
            target_path = os.path.join(PDF_DIR, uploaded_file.name)
            shutil.move(tmp_file_path, target_path)
            processed_files.append(target_path)
            
            st.success(f"Successfully processed: {uploaded_file.name}")
        else:
            st.error(f"Invalid file type: {uploaded_file.name}. Only PDF files are supported.")
    
    return processed_files

def create_rag_instance(document_paths=None):
    """Create and initialize LightRAG with documents."""
    embedding_dimension = 3072
    
    # Create a unique working directory for this batch of documents
    timestamp = int(time.time())
    working_dir = os.path.join(BASE_WORKING_DIR, f"rag_{timestamp}")
    os.makedirs(working_dir, exist_ok=True)
    
    # Initialize LightRAG instance
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension, 
            max_token_size=8192, 
            func=embedding_func
        ),
    )
    
    if document_paths:
        # Process and insert documents
        documents = process_documents(document_paths)
        if documents:
            logging.info(f"Processing {len(documents)} documents")
            for i, doc in enumerate(documents):
                logging.info(f"Processing document {i+1}/{len(documents)}")
                asyncio.run(async_insert_documents(rag, [doc]))
            
            # Save the RAG instance after processing all documents
            save_rag_instance(rag, STORAGE_DIR)
            logging.info("All documents processed and RAG instance saved")
            
            # Store the working directory in session state
            st.session_state.working_dirs.append(working_dir)
    
    return rag

def combine_rag_instances():
    """Combine multiple RAG instances into one."""
    if not st.session_state.working_dirs:
        return None
        
    # Create a new combined working directory
    timestamp = int(time.time())
    combined_dir = os.path.join(BASE_WORKING_DIR, f"combined_{timestamp}")
    os.makedirs(combined_dir, exist_ok=True)
    
    # Initialize combined RAG instance
    combined_rag = LightRAG(
        working_dir=combined_dir,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=3072,
            max_token_size=8192,
            func=embedding_func
        ),
    )
    
    # Combine all documents from previous instances
    for working_dir in st.session_state.working_dirs:
        try:
            # Copy all relevant files from previous instance
            for filename in os.listdir(working_dir):
                if any(sf in filename for sf in ['kv_store_', 'vdb_']):
                    src_path = os.path.join(working_dir, filename)
                    dst_path = os.path.join(combined_dir, filename)
                    if os.path.isfile(src_path):
                        shutil.copy2(src_path, dst_path)
                    elif os.path.isdir(src_path):
                        shutil.copytree(src_path, dst_path)
        except Exception as e:
            logging.error(f"Error combining RAG instance from {working_dir}: {e}")
    
    return combined_rag

def main():
    st.title("Legal Document RAG System")
    st.write("Upload PDF documents and ask questions about them")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    # Process uploaded files
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                processed_files = process_uploaded_files(uploaded_files)
                if processed_files:
                    st.session_state.uploaded_files.extend(processed_files)
                    new_rag = create_rag_instance(processed_files)
                    st.session_state.rag = combine_rag_instances()
                    st.success(f"Successfully processed {len(processed_files)} documents!")

if __name__ == "__main__":
    main()