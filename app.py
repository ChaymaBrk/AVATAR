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
    create_conversation_chain,
    create_reviewer_chain
)
from database import initialize_firebase
from ui import login_ui, register_ui, chat_ui

# Initialisation de Firebase
db = initialize_firebase()

# Initialisation des clients AWS
@st.cache_resource
def initialize_clients():
    bedrock_client = init_bedrock_client()
    s3_client = init_s3_client()
    return bedrock_client, s3_client

bedrock_client, s3_client = initialize_clients()

# Initialisation de l'état de session
def init_session_state():
    if "user" not in st.session_state:
        st.session_state.user = None
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "register" not in st.session_state:
        st.session_state.register = False
    if "show_history" not in st.session_state:
        st.session_state.show_history = False
    if "current_thread" not in st.session_state:
        st.session_state.current_thread = None
    
    # Chat states
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # AI components
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "reviewer" not in st.session_state:
        st.session_state.reviewer = None
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

# Chargement des ressources
@st.cache_resource(show_spinner="Initialisation de la base de connaissances juridiques...")
def load_resources(_s3_client):
    try:
        load_all_indices(_s3_client)
        faiss_indices = load_faiss_indices()
        combined_index = merge_indices(faiss_indices)
        return combined_index
    except Exception as e:
        st.error(f"Erreur de chargement: {str(e)}")
        return None

# Application principale
def main():
    init_session_state()
    
    if not st.session_state.resources_ready:
        with st.spinner("Chargement des ressources juridiques..."):
            combined_index = load_resources(s3_client)
            
            if combined_index:
                st.session_state.retriever = combined_index.as_retriever(
                    search_kwargs={"k": 3}
                )
                st.session_state.conversation, st.session_state.memory = create_conversation_chain(
                    st.session_state.retriever, 
                    bedrock_client
                )
                st.session_state.reviewer = create_reviewer_chain(bedrock_client)
                st.session_state.resources_ready = True
    
    if not st.session_state.get("logged_in"):
        if st.session_state.get("register"):
            register_ui()
        else:
            login_ui()
    else:
        if st.session_state.resources_ready:
            chat_ui(
                st.session_state.conversation,
                st.session_state.reviewer,
                st.session_state.memory
            )
        else:
            st.warning("Les ressources sont en cours de chargement...")
            st.spinner("Finalisation en cours...")

if __name__ == "__main__": 
    main()