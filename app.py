import streamlit as st

st.set_page_config(
        page_title="Legal Assistant Pro",
        page_icon="⚖️",
        layout="centered",
        initial_sidebar_state="auto"
    )

from services import (
    init_bedrock_client, 
    init_s3_client,
    load_all_indices,
    load_faiss_indices,
    merge_indices,
    create_conversation_chain
)
from database import init_db
from ui import login_ui, register_ui, chat_ui

# Initialize database
init_db()

# Initialize AWS clients
@st.cache_resource
def initialize_clients():
    bedrock_client = init_bedrock_client()
    s3_client = init_s3_client()
    return bedrock_client, s3_client

bedrock_client, s3_client = initialize_clients()

# Session state initialization
def init_session_state():
    # Authentication states
    if "username" not in st.session_state:
        st.session_state.username = None
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "register" not in st.session_state:
        st.session_state.register = False
    
    # Chat states
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # AI components
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "memory" not in st.session_state:
        from langchain.memory import ConversationBufferMemory
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )
    
    # Resource loading flags
    if "indices_loaded" not in st.session_state:
        st.session_state.indices_loaded = False
    if "resources_ready" not in st.session_state:
        st.session_state.resources_ready = False

# Load AI resources
@st.cache_resource(show_spinner="Initializing legal knowledge base...")
def load_resources(_s3_client):
    # Load documents from S3
    load_all_indices(_s3_client)
    
    # Load and merge FAISS indices
    faiss_indices = load_faiss_indices()
    combined_index = merge_indices(faiss_indices)
    
    if not combined_index:
        st.error("Failed to load legal resources. Please try again later.")
        return None
    
    return combined_index

# Main application logic
def main():
    # Initialize session state
    init_session_state()
    
    # Load resources if not already loaded
    if not st.session_state.resources_ready:
        with st.spinner("Loading legal resources..."):
            combined_index = load_resources(s3_client)
            
            if combined_index:
                # Initialize conversation chain
                st.session_state.retriever = combined_index.as_retriever(
                    search_kwargs={"k": 3}
                )
                st.session_state.conversation, st.session_state.memory = create_conversation_chain(
                    st.session_state.retriever, 
                    bedrock_client
                )
                st.session_state.resources_ready = True
    
    # Show appropriate UI based on authentication status
    if not st.session_state.get("logged_in"):
        if st.session_state.get("register"):
            register_ui()
        else:
            login_ui()
    else:
        if st.session_state.resources_ready:
            chat_ui(
                st.session_state.conversation, 
                st.session_state.memory
            )
        else:
            st.warning("Legal resources are still loading. Please wait...")
            st.spinner("Finalizing setup...")

# Run the application
if __name__ == "__main__":
   
    main()