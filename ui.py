import streamlit as st
from database import (
    firebase_sign_in, 
    firebase_sign_up,
    save_chat_history,
    get_user_history,
    get_user_threads
)
from config import APP_CONFIG
from langchain_core.messages import HumanMessage, AIMessage
from utils import detect_language
import time
from datetime import datetime

def set_white_blue_theme():
    """Simple white and blue theme for clean interface"""
    st.markdown(f"""
    <style>
        :root {{
            --primary: #1E88E5;  /* Main blue */
            --primary-light: #64B5F6;
            --primary-dark: #0D47A1;
            --secondary: #F5F5F5;  /* Light gray background */
            --accent: #2196F3;  /* Accent blue */
            --text-main: #212121;  /* Dark text */
            --text-secondary: #757575;
            --success: #4CAF50;
            --warning: #FF5722;
            --card: #FFFFFF;
            --card-light: #FAFAFA;
            --border: #E0E0E0;
        }}
        
        /* Clean white background */
        .stApp {{
            background-color: var(--secondary);
            color: var(--text-main);
            font-family: 'Roboto', 'Helvetica Neue', Arial, sans-serif;
        }}
        
        /* Import clean font */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
        
        /* Simple headings */
        h1, h2, h3 {{
            color: var(--primary-dark) !important;
            font-family: 'Roboto', sans-serif;
            font-weight: 500;
        }}
        
        h1 {{
            font-size: 2.5rem;
            margin: 1.5rem 0;
            border-bottom: 2px solid var(--primary-light);
            padding-bottom: 1rem;
        }}
        
        /* Sidebar styling */
        .css-1d391kg {{
            background-color: var(--card) !important;
            border-right: 1px solid var(--border);
        }}
        
        /* Input fields */
        .stTextInput input, .stTextArea textarea {{
            background-color: var(--card);
            color: var(--text-main);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 12px;
        }}
        
        .stTextInput input:focus, .stTextArea textarea:focus {{
            border-color: var(--primary);
            box-shadow: 0 0 0 2px rgba(30, 136, 229, 0.2);
        }}
        
        /* Buttons */
        .stButton>button {{
            background-color: var(--primary);
            color: white !important;
            border: none;
            border-radius: 4px;
            font-weight: 500;
            padding: 0.8rem 1.5rem;
            transition: all 0.3s ease;
        }}
        
        .stButton>button:hover {{
            background-color: var(--primary-dark);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        /* Chat messages */
        .stChatMessage {{
            border-radius: 8px;
            padding: 1.2rem;
            margin: 1rem 0;
            border: 1px solid var(--border);
        }}
        
        /* User message */
        [data-testid="stChatMessage-human"] {{
            background-color: var(--card-light);
            margin-right: 10%;
        }}
        
        /* AI message */
        [data-testid="stChatMessage-ai"] {{
            background-color: var(--card);
            margin-left: 10%;
            border-left: 3px solid var(--primary);
        }}
        
        /* Chat input */
        .stChatInputContainer {{
            background-color: var(--card);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid var(--border);
            margin-top: 2rem;
        }}
        
        /* Loading spinner */
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        .loading-spinner {{
            display: inline-block;
            width: 18px;
            height: 18px;
            border: 3px solid var(--primary-light);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 8px;
            vertical-align: middle;
        }}
    </style>
    """, unsafe_allow_html=True)

def simple_header():
    """Clean header with blue accent"""
    st.markdown("""
    <div style='text-align: center; margin: 2rem 0 3rem 0;'>
        <h1 style='margin-bottom: 0.5rem;'>
            <span style='color: var(--primary);'>Themis</span>
            <span style='color: var(--text-main);'>AI</span>
        </h1>
        <div style='color: var(--text-secondary); font-size: 1.1rem;'>
            Legal Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

def loading_indicator():
    """Simple loading indicator"""
    with st.empty():
        st.markdown("""
        <div style='text-align: center; color: var(--primary);'>
            <span class='loading-spinner'></span>
            <span>Processing...</span>
        </div>
        """, unsafe_allow_html=True)

def register_ui():
    """Clean registration interface"""
    set_white_blue_theme()
    simple_header()
    
    with st.container():
        st.subheader("Create Account")
        
        with st.form("register_form", border=False):
            cols = st.columns([1, 3, 1])
            with cols[1]:
                email = st.text_input("Email", 
                                    placeholder="your@email.com",
                                    key="reg_email")
                
                username = st.text_input("Username", 
                                       placeholder="Choose a username",
                                       key="reg_username")
                
                password = st.text_input("Password", 
                                       type="password", 
                                       placeholder="Minimum 8 characters",
                                       key="reg_password")
                
                confirm = st.text_input("Confirm Password", 
                                      type="password", 
                                      placeholder="Re-enter your password",
                                      key="reg_confirm")
                
                btn_cols = st.columns(2)
                with btn_cols[0]:
                    if st.form_submit_button("Create Account", use_container_width=True):
                        if email and username and password:
                            if password == confirm:
                                if len(password) >= 8:
                                    with st.spinner("Creating account..."):
                                        loading_indicator()
                                        try:
                                            uid = firebase_sign_up(email, password, username)
                                            st.success("Account created successfully!")
                                            time.sleep(1)
                                            st.session_state.register = False
                                            st.rerun()
                                        except Exception as e:
                                            st.error(str(e))
                                else:
                                    st.error("Password too weak (min 8 characters)")
                            else:
                                st.error("Passwords do not match")
                        else:
                            st.error("All fields required")
                
                with btn_cols[1]:
                    if st.form_submit_button("Back to Login", 
                                           use_container_width=True, 
                                           type="secondary"):
                        st.session_state.register = False
                        st.rerun()

def login_ui():
    """Clean login interface"""
    set_white_blue_theme()
    simple_header()
    
    with st.container():
        st.subheader("Login")
        
        with st.form("login_form", border=False):
            cols = st.columns([1, 3, 1])
            with cols[1]:
                email = st.text_input("Email", 
                                    placeholder="your@email.com",
                                    key="login_email")
                
                password = st.text_input("Password", 
                                       type="password", 
                                       placeholder="Enter your password",
                                       key="login_password")
                
                btn_cols = st.columns(2)
                with btn_cols[0]:
                    if st.form_submit_button("Login", use_container_width=True):
                        if email and password:
                            with st.spinner("Authenticating..."):
                                loading_indicator()
                                try:
                                    user = firebase_sign_in(email, password)
                                    st.session_state["user"] = user
                                    st.session_state["logged_in"] = True
                                    st.session_state["chat_history"] = []
                                    
                                    # Get default thread
                                    threads = get_user_threads(user['uid'])
                                    if threads:
                                        st.session_state["current_thread"] = next(iter(threads.keys()))
                                    
                                    st.success("Login successful")
                                    time.sleep(0.8)
                                    st.rerun()
                                except Exception as e:
                                    st.error(str(e))
                        else:
                            st.error("Email and password required")
                
                with btn_cols[1]:
                    if st.form_submit_button("Create Account", 
                                           use_container_width=True, 
                                           type="secondary"):
                        st.session_state.register = True
                        st.rerun()

def display_history_modal():
    """Clean history modal"""
    with st.container():
        st.markdown("""
        <div style='text-align: center; margin-bottom: 1.5rem;'>
            <h2 style='color: var(--primary);'>
                Conversation History
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.get("current_thread"):
            st.warning("No active thread selected")
            return
            
        history = get_user_history(
            st.session_state.user['uid'],
            st.session_state.current_thread
        )
        
        if not history:
            st.info("No history found")
            return
        
        for idx, (question, answer, timestamp) in enumerate(history, 1):
            try:
                dt = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp
                formatted_time = dt.strftime('%d/%m/%Y %H:%M')
            except:
                formatted_time = "Unknown date"
            
            with st.expander(f"Session #{idx} - {formatted_time}", expanded=False):
                st.markdown(f"""
                <div style='background-color: var(--card-light);
                            padding: 1.2rem; border-radius: 8px; margin-bottom: 1rem;
                            border-left: 3px solid var(--primary);'>
                    <div style='margin-bottom: 1rem;'>
                        <span style='color: var(--primary); font-weight: 500;'>Question:</span>
                        <div style='color: var(--text-main); padding: 0.5rem 0;'>{question}</div>
                    </div>
                    <div>
                        <span style='color: var(--primary); font-weight: 500;'>Response:</span>
                        <div style='color: var(--text-main); padding: 0.5rem 0;'>{answer}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        

def chat_ui(conversation_chain, reviewer_chain, memory):
    """Clean chat interface with history"""
    set_white_blue_theme()
    
    # Sidebar
    with st.sidebar:
        if 'user' in st.session_state:
            # Compact user profile
            st.markdown(f"""
            <div style='background-color: var(--card-light);
                        padding: 1rem; border-radius: 8px; margin-bottom: 1rem;
                        border-left: 2px solid var(--primary);'>
                <div style='display: flex; align-items: center;'>
                    <div style='margin-right: 1rem; color: var(--primary); font-size: 1.2rem;'>👤</div>
                    <div>
                        <div style='font-weight: 500; color: var(--text-main); line-height: 1.2;'>
                            {st.session_state.user['username']}
                        </div>
                        <div style='font-size: 0.8rem; color: var(--text-secondary);'>
                            {st.session_state.user['email']}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Compact button row
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("History", 
                        use_container_width=True,
                        help="View chat history",
                        key="history_btn"):
                st.session_state["show_history"] = True
        with col2:
            if st.button("Logout", 
                        use_container_width=True,
                        type="secondary",
                        help="Logout from account",
                        key="logout_btn"):
                st.session_state.clear()
                st.rerun()

        st.markdown("---")
    
    # Show history if activated
    if st.session_state.get("show_history"):
        display_history_modal()
        if st.button("← Back to Chat", use_container_width=True):
            st.session_state["show_history"] = False
            st.rerun()
        return
    
    # Main area
    simple_header()
    
    # Chat history
    for message in st.session_state.get("chat_history", []):
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(f"""
                <div style='display: flex; align-items: flex-start;'>
                    <div style='margin-right: 1rem; color: var(--primary);'>👤</div>
                    <div>
                        <div style='color: var(--text-main); line-height: 1.6;'>
                            {message.content}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            with st.chat_message("assistant"):
                st.markdown(f"""
                <div style='display: flex; align-items: flex-start;'>
                    <div style='margin-right: 1rem; color: var(--primary);'>⚖️</div>
                    <div>
                        <div style='color: var(--text-main); line-height: 1.6;'>
                            {message.content}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    query = st.chat_input("Enter your legal question...")
    if query and st.session_state.get("logged_in") and st.session_state.get("current_thread"):
        # Save user query
        st.session_state.chat_history.append(HumanMessage(content=query))
        
        # Show user message
        with st.chat_message("user"):
            st.markdown(f"""
            <div style='display: flex; align-items: flex-start;'>
                <div style='margin-right: 1rem; color: var(--primary);'>👤</div>
                <div>
                    <div style='color: var(--text-main); line-height: 1.6;'>
                        {query}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # System response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Stream response
            for chunk in conversation_chain.stream({
                "question": query,
                "chat_history": st.session_state.chat_history,
                "language": detect_language(query)
            }):
                full_response += chunk
                response_placeholder.markdown(f"""
                <div style='display: flex; align-items: flex-start;'>
                    <div style='margin-right: 1rem; color: var(--primary);'>⚖️</div>
                    <div>
                        <div style='color: var(--text-main); line-height: 1.6;'>
                            {full_response}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.session_state.chat_history.append(AIMessage(content=full_response))
            
            # Save to Firebase
            save_chat_history(
                st.session_state.user['uid'],
                st.session_state.current_thread,
                query,
                full_response
            )
        
        memory.save_context({"question": query}, {"output": full_response})