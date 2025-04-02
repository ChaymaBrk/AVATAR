import os
import json
import asyncio

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

from dotenv import load_dotenv
import logging
from openai import AzureOpenAI
import numpy as np

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

# Storage directories
WORKING_DIR = "./legal_documents"
STORAGE_DIR = "./rag_storage"

async def llm_model_func(
    prompt, 
    system_prompt=None, 
    history_messages=[], 
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

async def embedding_func(texts: list[str]) -> np.ndarray:
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

def load_rag_instance():
    """Load previously saved RAG configuration and create LightRAG instance."""
    # Load configuration
    with open(os.path.join(STORAGE_DIR, 'rag_config.json'), 'r') as f:
        config = json.load(f)
    
    # Recreate LightRAG instance with saved configuration
    rag = LightRAG(
        working_dir=config['working_dir'],
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=config['embedding_func_config']['embedding_dim'],
            max_token_size=config['embedding_func_config']['max_token_size'],
            func=embedding_func
        ),
    )
    
    return rag

def main():
    # Load the RAG instance
    rag = load_rag_instance()
    
    # Interactive query loop
    while True:
        query_text = input("\nEnter your legal query (or 'exit' to quit): ")
        
        if query_text.lower() == 'exit':
            break
        
        print("\nQuerying documents...")
        
        # Demonstrate different query modes
        print("Result (Naive):")
        print(rag.query(query_text, param=QueryParam(mode="naive")))
        
        print("\nResult (Local):")
        print(rag.query(query_text, param=QueryParam(mode="local")))
        
        print("\nResult (Global):")
        print(rag.query(query_text, param=QueryParam(mode="global")))
        
        print("\nResult (Hybrid):")
        print(rag.query(query_text, param=QueryParam(mode="hybrid")))

if __name__ == "__main__":
    main()