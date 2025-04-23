import streamlit as st
from database import (
    firebase_sign_in, 
    firebase_sign_up,
    save_chat_history,
    get_user_history,
    clear_user_history,
    get_user_threads
)
from config import APP_CONFIG
from langchain_core.messages import HumanMessage, AIMessage
from utils import detect_language
import time
from datetime import datetime

def set_neon_legal_theme():
    """Thème néo-juridique futuriste pour domination du marché"""
    st.markdown(f"""
    <style>
        :root {{
            --primary-dark: #0F1A3F;  /* Bleu nocturne */
            --primary: #1D2B64;
            --primary-light: #2C3B8B;
            --secondary: #0A0E1A;  /* Fond spatial */
            --accent: #00D4FF;  /* Cyan néon */
            --accent-light: #6BFFFF;
            --text-main: #E6F1FF;  /* Blanc stellaire */
            --text-secondary: #8A9EFF;
            --success: #00FFA3;
            --warning: #FF2D75;
            --card-dark: #121A33;
            --card: #1A2347;
            --card-light: #232D5A;
            --transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            --neon-glow: 0 0 8px rgba(0, 212, 255, 0.8);
        }}
        
        /* Fond immersif */
        .stApp {{
            background: radial-gradient(ellipse at center, var(--secondary) 0%, #050813 100%);
            color: var(--text-main);
            font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
            min-height: 100vh;
        }}
        
        /* Police high-tech */
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&family=Inter:wght@400;600&display=swap');
        
        /* Titres futuristes */
        h1, h2, h3 {{
            text-align: center !important;
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 0.05em;
            text-shadow: var(--neon-glow);
        }}
        
        h1 {{
            color: var(--accent) !important;
            font-size: 3rem;
            margin: 2rem 0;
            background: linear-gradient(90deg, var(--accent), var(--accent-light));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        h2 {{
            color: var(--text-main) !important;
            border-bottom: 2px solid var(--primary-light);
            padding-bottom: 1rem;
            margin: 2.5rem auto;
            width: 80%;
            font-size: 2rem;
            position: relative;
        }}
        
        h2:after {{
            content: "";
            position: absolute;
            bottom: -2px;
            left: 25%;
            width: 50%;
            height: 2px;
            background: var(--accent);
            box-shadow: var(--neon-glow);
        }}
        
        /* Sidebar cyber */
        .css-1d391kg {{
            background: linear-gradient(180deg, var(--card-dark) 0%, var(--secondary) 100%) !important;
            border-right: 1px solid var(--primary-light);
            box-shadow: 8px 0 20px rgba(0,0,0,0.4);
        }}
        
        /* Inputs futuristes */
        .stTextInput input, .stTextArea textarea {{
            background-color: rgba(26, 35, 71, 0.7);
            color: var(--text-main);
            border: 1px solid var(--primary-light);
            border-radius: 8px;
            padding: 14px;
            transition: var(--transition);
            backdrop-filter: blur(5px);
        }}
        
        .stTextInput input:focus, .stTextArea textarea:focus {{
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.3);
            background-color: rgba(26, 35, 71, 0.9);
        }}
        
        /* Boutons cybernétiques */
        .stButton>button {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
            color: white !important;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            padding: 1rem 2rem;
            transition: var(--transition);
            box-shadow: 0 4px 15px rgba(29, 43, 100, 0.4);
            position: relative;
            overflow: hidden;
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            font-size: 0.9rem;
        }}
        
        .stButton>button:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0, 212, 255, 0.4);
        }}
        
        .stButton>button:active {{
            transform: translateY(0);
        }}
        
        .stButton>button:before {{
            content: "";
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, 
                          transparent, 
                          rgba(0, 212, 255, 0.2), 
                          transparent);
            transition: 0.5s;
        }}
        
        .stButton>button:hover:before {{
            left: 100%;
        }}
        
        /* Messages de chat holographiques */
        .stChatMessage {{
            border-radius: 12px;
            padding: 1.8rem;
            margin: 1.5rem 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            position: relative;
            overflow: hidden;
            border: none;
            backdrop-filter: blur(5px);
        }}
        
        /* Message utilisateur */
        [data-testid="stChatMessage-human"] {{
            background: linear-gradient(135deg, 
                          rgba(15, 26, 63, 0.8) 0%, 
                          rgba(44, 59, 139, 0.6) 100%);
            border-left: 4px solid var(--accent);
            margin-right: 15%;
            border-top-right-radius: 0;
        }}
        
        /* Message AI */
        [data-testid="stChatMessage-ai"] {{
            background: linear-gradient(135deg, 
                          rgba(26, 35, 71, 0.8) 0%, 
                          rgba(35, 45, 90, 0.6) 100%);
            border-left: 4px solid var(--primary-light);
            margin-left: 15%;
            border-top-left-radius: 0;
            box-shadow: 0 0 15px rgba(0, 212, 255, 0.2);
        }}
        
        /* Zone de saisie futuriste */
        .stChatInputContainer {{
            background: rgba(18, 26, 51, 0.7);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid var(--primary-light);
            margin-top: 3rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(5px);
        }}
        
        /* Animation de chargement cyber */
        @keyframes cyber-spin {{
            0% {{ transform: rotate(0deg); opacity: 0.6; }}
            50% {{ opacity: 1; }}
            100% {{ transform: rotate(360deg); opacity: 0.6; }}
        }}
        
        .cyber-loading {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid var(--accent);
            border-radius: 50%;
            border-top-color: transparent;
            animation: cyber-spin 1s linear infinite;
            margin-right: 10px;
            vertical-align: middle;
        }}
        
        /* Effet néon */
        .neon-text {{
            text-shadow: 0 0 5px var(--accent), 
                         0 0 10px var(--accent), 
                         0 0 20px var(--accent);
            animation: neon-pulse 1.5s infinite alternate;
        }}
        
        @keyframes neon-pulse {{
            from {{ opacity: 0.8; }}
            to {{ opacity: 1; }}
        }}
    </style>
    """, unsafe_allow_html=True)

def cyber_header():
    """En-tête cybernétique avec effets visuels avancés"""
    st.markdown("""
    <div style='text-align: center; margin: 3rem 0 4rem 0; position: relative;'>
        <div style='position: absolute; 
                    top: 50%; left: 50%; 
                    transform: translate(-50%, -50%);
                    width: 300px; height: 300px;
                    background: radial-gradient(circle, rgba(0,212,255,0.15) 0%, transparent 70%);
                    z-index: 0;'>
        </div>
        <h1 style='position: relative; z-index: 2; margin-bottom: 0.5rem;'>
            <span style='color: var(--accent);'>Themis</span>
            <span style='color: var(--text-main);'>AI</span>
            <span style='color: var(--accent);'></span>
        </h1>
        <div style='position: relative; z-index: 2;'>
            <span style='color: var(--text-secondary); font-size: 1.2rem; letter-spacing: 0.2em;'>
                LEGAL INTELLIGENCE PLATFORM
            </span>
        </div>
        <div style='position: absolute; bottom: -20px; left: 50%; transform: translateX(-50%);
                    width: 200px; height: 2px;
                    background: linear-gradient(90deg, transparent, var(--accent), transparent);
                    box-shadow: 0 0 10px var(--accent);'>
        </div>
    </div>
    """, unsafe_allow_html=True)

def cyber_loading():
    """Animation de chargement cybernétique"""
    with st.empty():
        for i in range(3):
            dots = "•" * (i + 1)
            st.markdown(f"""
            <div style='text-align: center; color: var(--accent); font-family: "Orbitron", sans-serif;'>
                <span class='cyber-loading'></span>
                <span>INITIALIZING LEGAL MODULE{dots}</span>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.4)
        st.markdown("""
        <div style='text-align: center; color: var(--success); font-family: "Orbitron", sans-serif;'>
            <span>SYSTEM READY</span>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.3)

def register_ui():
    """Interface d'inscription cybernétique"""
    set_neon_legal_theme()
    cyber_header()
    
    with st.container():
        st.subheader("CREATE ACCOUNT")
        
        with st.form("register_form", border=False):
            cols = st.columns([1, 3, 1])
            with cols[1]:
                st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                
                email = st.text_input("ENTER EMAIL", 
                                    placeholder="e.g. legal@firm.com",
                                    key="reg_email")
                
                username = st.text_input("ENTER USERNAME", 
                                       placeholder="e.g. FIRM_ACCESS_01",
                                       key="reg_username")
                
                password = st.text_input("SET SECURITY KEY", 
                                       type="password", 
                                       placeholder="12 CHARACTER MINIMUM",
                                       key="reg_password")
                
                confirm = st.text_input("CONFIRM SECURITY KEY", 
                                      type="password", 
                                      placeholder="RE-ENTER YOUR KEY",
                                      key="reg_confirm")
                
                st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
                
                btn_cols = st.columns(2)
                with btn_cols[0]:
                    if st.form_submit_button("🚀 ACTIVATE ACCOUNT", use_container_width=True):
                        if email and username and password:
                            if password == confirm:
                                if len(password) >= 8:
                                    with st.spinner("SECURING YOUR ACCESS..."):
                                        cyber_loading()
                                        try:
                                            uid = firebase_sign_up(email, password, username)
                                            st.success("ACCOUNT CREATED SUCCESSFULLY!")
                                            time.sleep(1)
                                            st.session_state.register = False
                                            st.rerun()
                                        except Exception as e:
                                            st.error(str(e))
                                else:
                                    st.error("SECURITY KEY TOO WEAK (MIN 8 CHARACTERS)")
                            else:
                                st.error("SECURITY KEYS DO NOT MATCH")
                        else:
                            st.error("ALL FIELDS REQUIRED")
                
                with btn_cols[1]:
                    if st.form_submit_button("← BACK TO LOGIN", 
                                           use_container_width=True, 
                                           type="secondary"):
                        st.session_state.register = False
                        st.rerun()

def login_ui():
    """Interface de connexion cybernétique"""
    set_neon_legal_theme()
    cyber_header()
    
    with st.container():
        st.subheader("SECURE ACCESS")
        
        with st.form("login_form", border=False):
            cols = st.columns([1, 3, 1])
            with cols[1]:
                st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                
                email = st.text_input("EMAIL", 
                                    placeholder="ENTER YOUR EMAIL",
                                    key="login_email")
                
                password = st.text_input("SECURITY KEY", 
                                       type="password", 
                                       placeholder="ENTER YOUR KEY",
                                       key="login_password")
                
                st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
                
                btn_cols = st.columns(2)
                with btn_cols[0]:
                    if st.form_submit_button("🔐 AUTHENTICATE", use_container_width=True):
                        if email and password:
                            with st.spinner("VERIFYING CREDENTIALS..."):
                                cyber_loading()
                                try:
                                    user = firebase_sign_in(email, password)
                                    st.session_state["user"] = user
                                    st.session_state["logged_in"] = True
                                    st.session_state["chat_history"] = []
                                    
                                    # Récupérer le premier thread par défaut
                                    threads = get_user_threads(user['uid'])
                                    if threads:
                                        st.session_state["current_thread"] = next(iter(threads.keys()))
                                    
                                    st.success("ACCESS GRANTED")
                                    time.sleep(0.8)
                                    st.rerun()
                                except Exception as e:
                                    st.error(str(e))
                        else:
                            st.error("FIELDS REQUIRED")
                
                with btn_cols[1]:
                    if st.form_submit_button("🆕 NEW ACCOUNT", 
                                           use_container_width=True, 
                                           type="secondary"):
                        st.session_state.register = True
                        st.rerun()

def display_history_modal():
    """Modal d'historique dans le style cybernétique"""
    with st.container():
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h2 style='color: var(--accent);'>
                <span style='font-family: "Orbitron", sans-serif;'>CONVERSATION HISTORY</span>
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
            st.markdown("""
            <div style='text-align: center; color: var(--text-secondary); font-family: "Orbitron", sans-serif;'>
                NO HISTORY FOUND
            </div>
            """, unsafe_allow_html=True)
            return
        
        for idx, (question, answer, timestamp) in enumerate(history, 1):
            try:
                dt = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp
                formatted_time = dt.strftime('%d/%m/%Y %H:%M')
            except:
                formatted_time = "Unknown date"
            
            with st.expander(f"SESSION #{idx} - {formatted_time}", expanded=False):
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, var(--card-dark) 0%, var(--card) 100%);
                            padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;
                            border-left: 3px solid var(--accent);'>
                    <div style='margin-bottom: 1rem;'>
                        <span style='color: var(--accent-light); font-weight: 600; font-family: "Orbitron", sans-serif;'>QUERY:</span>
                        <div style='color: var(--text-main); padding: 0.5rem 0;'>{question}</div>
                    </div>
                    <div>
                        <span style='color: var(--accent-light); font-weight: 600; font-family: "Orbitron", sans-serif;'>RESPONSE:</span>
                        <div style='color: var(--text-secondary); padding: 0.5rem 0;'>{answer}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        if st.button("🗑️ PURGE HISTORY", type="secondary"):
            if clear_user_history(
                st.session_state.user['uid'],
                st.session_state.current_thread
            ):
                st.success("HISTORY CLEARED SUCCESSFULLY!")
                time.sleep(1)
                st.rerun()

def chat_ui(conversation_chain, memory):
    """Interface de consultation cybernétique avec historique"""
    set_neon_legal_theme()
    
    # Sidebar futuriste
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 3rem;'>
            <div style='font-size: 2.5rem; color: var(--accent);'>⚡</div>
            <h3 style='color: var(--accent-light); margin-top: 0.5rem;'>JURISAIX CORE</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if 'user' in st.session_state:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, var(--card-dark) 0%, var(--card) 100%);
                        padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;
                        border-left: 3px solid var(--accent);
                        box-shadow: 0 0 15px rgba(0, 212, 255, 0.1);'>
                <div style='display: flex; align-items: center;'>
                    <div style='font-size: 1.8rem; margin-right: 1rem; color: var(--accent);'>👨‍⚖️</div>
                    <div>
                        <div style='font-weight: 600; color: var(--text-main); font-size: 1.1rem;'>
                            {st.session_state.user['username']}
                        </div>
                        <div style='font-size: 0.8rem; color: var(--text-secondary); letter-spacing: 0.05em;'>
                            PREMIUM ACCESS • VERIFIED
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Gestion des threads
        if 'user' in st.session_state:
            threads = get_user_threads(st.session_state.user['uid'])
            if threads:
                st.subheader("YOUR THREADS")
                for thread_id, thread_data in threads.items():
                    if st.button(
                        thread_data.get('thread_name', 'Unnamed Thread'),
                        key=f"thread_{thread_id}",
                        use_container_width=True
                    ):
                        st.session_state.current_thread = thread_id
                        st.rerun()
        
        # Bouton historique
        if st.button("📜 VIEW HISTORY", 
                    use_container_width=True,
                    key="history_btn"):
            st.session_state["show_history"] = True
        
        if st.button("⏻ TERMINATE SESSION", 
                    use_container_width=True,
                    key="logout_btn"):
            st.session_state.clear()
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("""
        <div style='margin-bottom: 2rem;'>
            <h4 style='color: var(--accent-light); margin-bottom: 1rem; 
                        font-family: "Orbitron", sans-serif; letter-spacing: 0.05em;'>
                📡 KNOWLEDGE MODULES
            </h4>
            <div style='background: linear-gradient(135deg, var(--card) 0%, var(--card-light) 100%);
                        padding: 1.2rem; border-radius: 10px;
                        box-shadow: 0 0 10px rgba(0, 212, 255, 0.1);'>
                <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
                    <div style='margin-right: 1rem; font-size: 1.2rem;'>🌐</div>
                    <div>GLOBAL LEGISLATION</div>
                </div>
                <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
                    <div style='margin-right: 1rem; font-size: 1.2rem;'>🔍</div>
                    <div>PRECEDENT ANALYSIS</div>
                </div>
                <div style='display: flex; align-items: center;'>
                    <div style='margin-right: 1rem; font-size: 1.2rem;'>⚖️</div>
                    <div>JURISDICTIONAL AI</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div style='text-align: center; color: var(--text-secondary); font-size: 0.7rem; 
                    margin-top: 3rem; letter-spacing: 0.05em;'>
            <div style='margin-bottom: 0.5rem;'>JURISAIX LEGAL TECHNOLOGY</div>
            <div>v4.2.1 • QUANTUM ENCRYPTION • © 2024</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Affichage de l'historique si activé
    if st.session_state.get("show_history"):
        display_history_modal()
        if st.button("← BACK TO CHAT", use_container_width=True):
            st.session_state["show_history"] = False
            st.rerun()
        return
    
    # Zone principale
    cyber_header()
    
    # Historique de consultation
    for message in st.session_state.get("chat_history", []):
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(f"""
                <div style='display: flex; align-items: flex-start; margin-bottom: 0.8rem;'>
                    <div style='font-size: 1.5rem; margin-right: 1rem; color: var(--accent);'>👤</div>
                    <div>
                        <div style='font-weight: 600; color: var(--accent); 
                                    font-family: "Orbitron", sans-serif; letter-spacing: 0.05em;
                                    margin-bottom: 0.5rem;'>
                            USER QUERY
                        </div>
                        <div style='color: var(--text-main); line-height: 1.6;'>
                            {message.content}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            with st.chat_message("assistant"):
                st.markdown(f"""
                <div style='display: flex; align-items: flex-start; margin-bottom: 0.8rem;'>
                    <div style='font-size: 1.5rem; margin-right: 1rem; color: var(--accent-light);'>⚡</div>
                    <div>
                        <div style='font-weight: 600; color: var(--accent-light); 
                                    font-family: "Orbitron", sans-serif; letter-spacing: 0.05em;
                                    margin-bottom: 0.5rem;'>
                            JURISAIX RESPONSE
                        </div>
                        <div style='color: var(--text-main); line-height: 1.6;'>
                            {message.content}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Interface de requête
    query = st.chat_input("ENTER LEGAL QUERY...")
    if query and st.session_state.get("logged_in") and st.session_state.get("current_thread"):
        # Enregistrement de la requête
        st.session_state.chat_history.append(HumanMessage(content=query))
        
        # Affichage de la requête utilisateur
        with st.chat_message("user"):
            st.markdown(f"""
            <div style='display: flex; align-items: flex-start; margin-bottom: 0.8rem;'>
                <div style='font-size: 1.5rem; margin-right: 1rem; color: var(--accent);'>👤</div>
                <div>
                    <div style='font-weight: 600; color: var(--accent); 
                                font-family: "Orbitron", sans-serif; letter-spacing: 0.05em;
                                margin-bottom: 0.5rem;'>
                        USER QUERY
                    </div>
                    <div style='color: var(--text-main); line-height: 1.6;'>
                        {query}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Réponse du système avec effet de streaming
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Animation pendant le chargement
            with st.spinner(""):
                cyber_loading()
            
            # Simulation de streaming
            for chunk in conversation_chain.stream({
                "question": query,
                "chat_history": st.session_state.chat_history,
                "language": detect_language(query)
            }):
                full_response += chunk
                response_placeholder.markdown(f"""
                <div style='display: flex; align-items: flex-start; margin-bottom: 0.8rem;'>
                    <div style='font-size: 1.5rem; margin-right: 1rem; color: var(--accent-light);'>⚡</div>
                    <div>
                        <div style='font-weight: 600; color: var(--accent-light); 
                                    font-family: "Orbitron", sans-serif; letter-spacing: 0.05em;
                                    margin-bottom: 0.5rem;'>
                            JURISAIX RESPONSE
                        </div>
                        <div style='color: var(--text-main); line-height: 1.6;'>
                            {full_response}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.session_state.chat_history.append(AIMessage(content=full_response))
            
            # Sauvegarde dans Firebase
            save_chat_history(
                st.session_state.user['uid'],
                st.session_state.current_thread,
                query,
                full_response
            )
        
        memory.save_context({"question": query}, {"output": full_response})