import streamlit as st
import boto3
import os
import sqlite3
import time
from functools import wraps
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import HumanMessage, AIMessage   
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langdetect import detect
from botocore.config import Config
from diskcache import Cache
import concurrent.futures
from datetime import datetime, timezone

# Décorateur pour limiter le taux de requêtes
def rate_limited(max_per_second):
    min_interval = 1.0 / max_per_second
    def decorate(func):
        last_time_called = 0.0
        @wraps(func)
        def rate_limited_function(*args, **kwargs):
            nonlocal last_time_called
            elapsed = time.time() - last_time_called
            wait = min_interval - elapsed
            if wait > 0:
                time.sleep(wait)
            last_time_called = time.time()
            return func(*args, **kwargs)
        return rate_limited_function
    return decorate

# Décorateur pour mesurer le temps d'exécution
def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        st.sidebar.write(f"⏱️ {func.__name__} executed in: {end_time - start_time:.2f}s")
        return result
    return wrapper

# Configuration des variables d'environnement
os.environ['AWS_ACCESS_KEY_ID'] = 'AKIATWBJZ4G6IN7JILVW'
os.environ['AWS_SECRET_ACCESS_KEY'] = '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'
os.environ['AWS_DEFAULT_REGION'] = 'eu-central-1'
os.environ['BUCKET_NAME'] = 'kb-pdf'

# Configuration du client Bedrock avec backoff exponentiel
bedrock_config = Config(
    retries={
        'max_attempts': 10,
        'mode': 'adaptive',
    }
)

# Initialisation des clients AWS
@timeit
def init_aws_clients():
    return boto3.client(
        service_name='bedrock-runtime',
        region_name='eu-central-1',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        config=bedrock_config
    ), boto3.client('s3')

bedrock_client, s3_client = init_aws_clients()
BUCKET_NAME = "kb-pdf"

# Initialisation des embeddings
@timeit
def init_embeddings():
    return BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock_client)

bedrock_embeddings = init_embeddings()

# Initialisation du cache
cache = Cache('llm_cache')

# Configuration des dossiers
folder_path = "local_data/" 
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Initialisation de la base de données SQLite
@timeit
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS threads
                 (thread_id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, thread_name TEXT)''')
    conn.commit()
    conn.close()

@timeit
def add_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        st.success("User registered successfully!")
        create_thread(username, "Default Thread")
    except sqlite3.IntegrityError:
        st.error("Username already exists. Please choose a different username.")
    finally:
        conn.close()

@timeit
def authenticate(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    result = c.fetchone()
    conn.close()
    return result is not None

@timeit
def create_thread(username, thread_name):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT INTO threads (username, thread_name) VALUES (?, ?)", (username, thread_name))
    thread_id = c.lastrowid
    conn.commit()
    conn.close()
    return thread_id

@timeit
def get_threads(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT thread_id, thread_name FROM threads WHERE username = ?", (username,))
    threads = c.fetchall()
    conn.close()
    return threads

@rate_limited(5)  # Limite à 5 requêtes par seconde
@timeit
def get_llm(model_id="anthropic.claude-3-haiku-20240307-v1:0", temperature=0.3):
    return ChatBedrock(
        model_id=model_id,
        client=bedrock_client,
        temperature=temperature,
        model_kwargs={
            'max_tokens': 2048
        }
    )

def is_legal_query(query):
    """Détermine si la requête est juridique"""
    query_lower = query.lower().strip()
    
    # Liste des mots-clés juridiques
    legal_keywords = [
        "article", "loi", "code", "droit", "juridique", "contrat", 
        "tribunal", "jugement", "procédure", "litige", "divorce",
        "héritage", "propriété", "locatif", "pénal", "civil", "commercial",
        "tunisie", "tunisien", "jurisprudence", "avocat", "avocate", "juger"
    ]
    
    # Vérifier la présence de mots-clés juridiques ou de questions
    return any(keyword in query_lower for keyword in legal_keywords) or "?" in query

@cache.memoize(expire=3600)  # Cache pour 1 heure
@timeit
def rewrite_legal_query(original_query, chat_history=[]):
    """Réécrit uniquement les requêtes juridiques complexes"""
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", """Vous êtes un expert en réécriture de requêtes juridiques. 
        Votre tâche est uniquement de reformuler la requête pour qu'elle soit plus concise (max 20 mots) 
        tout en conservant son sens juridique précis, sans jamais répondre à la question.
        
        Format de sortie strict: "Requête réécrite: [version réécrite]"
        
        Exemple:
        Requête originale: "Quelles sont les conditions légales pour divorcer en Tunisie lorsque l'un des époux est étranger?"
        Requête réécrite: "Divorce en Tunisie avec époux étranger"
        
        Ne donnez jamais de conseils ou de réponse, seulement la reformulation."""),
        *chat_history,
        ("human", "{query}")
    ])
    
    rewriter_chain = (
        {"query": RunnablePassthrough()}
        | rewrite_prompt
        | get_llm(model_id="anthropic.claude-3-haiku-20240307-v1:0", temperature=0.1)
        | StrOutputParser()
    )
    
    return rewriter_chain.invoke(original_query)

@rate_limited(3)  # Limite à 3 requêtes par seconde
@timeit
def get_response(llm, faiss_index, question, chat_history=[]):
    """Obtient une réponse en utilisant le RAG avec query rewriting pour les requêtes juridiques"""
    # Vérifier si c'est un remerciement
    if question.lower().strip() in ["merci", "thank you", "شكرا"]:
        return "Je vous en prie. N'hésitez pas si vous avez d'autres questions juridiques."
    
    # Vérifier si c'est une salutation
    if question.lower().strip() in ["bonjour", "hello", "salut", "hi"]:
        return "Bonjour ! Je suis votre assistant juridique. Posez-moi vos questions sur le droit tunisien."
    
    prompt_template = """
    ## 🔹 **Role**:
    You are a retriever. Your task is to **retrieve relevant legal articles** based on the user's question. You must provide **clear, precise, and professional responses** that directly address the user's legal query.

    2. **Relevance**:
       - **Only return articles that directly address the user's question or problem.**
       - **Do NOT return irrelevant or approximate articles.**

    3. **Response Format**:
       - Return only the **raw text** of the retrieved article(s), without any comments or explanations.
       - If **no relevant article** is found, respond clearly:
         > "No precise legal information found. Please provide more details about your situation."

    Context: {context}  
    Question: {question}"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    answer = qa.invoke({"query": question})
    return answer['result']

@timeit
def create_conversation_chain(retriever):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """## 🔹 **Role**:
You are a **Tunisian lawyer**. Your task is to provide **clear, precise, and professional answers** based solely on the retrieved legal articles. You must help the user understand their legal situation, providing relevant explanations, clarifications, and examples when necessary.

## 🔹 **Guidelines**:
1. **Answer directly to the user's question**:
   - Don't start with phrases like "According to the context provided".
   - Provide a **concise, direct answer** for straightforward or factual questions (e.g., referencing a specific article of law).
   - For broader or more complex questions (e.g., interpretations of legal principles or explanations), offer a **detailed, well-structured response** with examples, legal references, or relevant case law where applicable.
   
2. **Level of Detail**:
   - **If the question is broad, complex, or requires interpretation** (e.g., a law principle), give a **detailed explanation** that includes relevant context, legal examples, or case law where applicable.
   - **If the question is straightforward** (e.g., referring to a specific article of law), give a **concise, direct answer** with minimal elaboration.

3. **Legal Precision**:
   - Ensure all responses are **legally accurate**, clear, and free from ambiguity.
   - When providing legal interpretations, clarify whether the response is based on the **specific context** or **general legal principles**.
   
4. **Ethical Responsibility**:
   - If the question concerns **personal or sensitive legal matters**, always remind the user to **consult a qualified lawyer**.

5. **Formatting**:
   - **Use clear and professional language**. Avoid colloquial or casual speech.
   - **Structure your responses for readability**:
     - Use short paragraphs.
     - Where applicable, use **numbered or bullet-point lists** for clarity.
   
6. **Language**:
   - **Always reply in the same language used by the user: {language}**, and **formally**
   - The response must be **formal**, professional, and precise.

7. **Clarification**:
   - If the provided legal articles do not fully answer the question, **politely ask the user for further details** or clarification.
  
8. **Acknowledgements**:
   - For simple acknowledgements like "thank you", respond briefly and professionally.
   - Example: "Je vous en prie. N'hésitez pas si vous avez d'autres questions juridiques."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = (
        RunnablePassthrough.assign(
            chat_history=lambda x: memory.load_memory_variables({})["chat_history"],
            language=lambda x: detect(x["question"]) if "question" in x else "fr"
        )
        | prompt_template
        | get_llm()
        | StrOutputParser()
    )

    return chain, memory

@timeit
def load_all_indices():
    # Vérifier si le dossier local existe
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Obtenir la liste des fichiers distants
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME)
    if 'Contents' not in response:
        st.error("No files found in the bucket.")
        return None

    # Filtrer seulement les fichiers FAISS/PKL
    s3_files = {obj['Key']: obj['LastModified'].replace(tzinfo=None) for obj in response['Contents'] 
                if obj['Key'].endswith(('.faiss', '.pkl'))}

    # Fonction pour télécharger un fichier
    def download_file(key):
        local_path = os.path.join(folder_path, key)
        
        # Ne télécharger que si le fichier n'existe pas ou est obsolète
        if not os.path.exists(local_path) or \
           (key in s3_files and 
            datetime.fromtimestamp(os.path.getmtime(local_path)).replace(tzinfo=None) < s3_files[key]):
            
            s3_client.download_file(
                Filename=local_path,
                Bucket=BUCKET_NAME,
                Key=key
            )
            return f"Downloaded {key}"
        return f"Skipped {key} (up to date)"

    # Téléchargement parallèle avec ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(download_file, s3_files.keys()))
    
    st.sidebar.write("\n".join(results))

@timeit
def load_faiss_indices():
    dir_list = os.listdir(folder_path)
    faiss_indices = []
    for file in dir_list:
        if file.endswith('.faiss'):
            index_name = file[:-6]
            try:
                faiss_index = FAISS.load_local(
                    index_name=index_name,
                    folder_path=folder_path,
                    embeddings=bedrock_embeddings,
                    allow_dangerous_deserialization=True
                )
                faiss_indices.append(faiss_index)
            except Exception as e:
                st.error(f"Failed to load index {index_name}: {e}")
    return faiss_indices

@timeit
def merge_indices(faiss_indices):
    if not faiss_indices:
        st.error("No valid FAISS indices found.")
        return None

    combined_index = faiss_indices[0]
    for index in faiss_indices[1:]:
        combined_index.merge_from(index)
    return combined_index

def register():
    st.title("Register")
    username = st.text_input("Choose a username")
    password = st.text_input("Choose a password", type="password")
    if st.button("Register"):
        if username and password:
            add_user(username, password)
        else:
            st.error("Please fill in all fields.")

def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            st.session_state["username"] = username
            st.session_state["logged_in"] = True
            threads = get_threads(username)
            if threads:
                st.session_state["current_thread"] = threads[0][0]
                st.session_state["chat_history"] = []
            st.success("Logged in successfully!")
            st.rerun()
        else:
            st.error("Invalid username or password")

def main():
    st.header("Tunisian Legal Assistant with Memory and Query Rewriting")

    # Initialisation de l'état de la session
    if "username" not in st.session_state:
        st.session_state["username"] = None
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "current_thread" not in st.session_state:
        st.session_state["current_thread"] = None
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "retriever" not in st.session_state:
        st.session_state["retriever"] = None
    if "conversation" not in st.session_state:
        st.session_state["conversation"] = None
    if "memory" not in st.session_state:
        st.session_state["memory"] = None
    if "original_question" not in st.session_state:
        st.session_state["original_question"] = ""
    if "rewritten_question" not in st.session_state:
        st.session_state["rewritten_question"] = ""

    # Chargement optimisé des indices
    if 'indices_loaded' not in st.session_state:
        load_all_indices()
        st.session_state.indices_loaded = True
    
    faiss_indices = load_faiss_indices()
    combined_index = merge_indices(faiss_indices)
    
    if not combined_index:
        return

    # Initialiser le retriever et la conversation
    if st.session_state.retriever is None:
        st.session_state["retriever"] = combined_index.as_retriever(search_kwargs={"k": 3})
        st.session_state["conversation"], st.session_state["memory"] = create_conversation_chain(st.session_state["retriever"])

    # Afficher l'historique de chat
    for message in st.session_state.chat_history:
        with st.chat_message("Human" if isinstance(message, HumanMessage) else "AI"):
            st.markdown(message.content)

    # Input de chat
    query = st.chat_input("Ask a question about Tunisian law")
    if query and st.session_state.logged_in and st.session_state.current_thread:
        # Gestion des salutations et remerciements
        if query.lower().strip() in ["bonjour", "hello", "salut", "hi"]:
            st.session_state.chat_history.append(HumanMessage(content=query))
            st.session_state.chat_history.append(AIMessage(content="Bonjour ! Je suis votre assistant juridique. Posez-moi vos questions sur le droit tunisien."))
            with st.chat_message("AI"):
                st.markdown("Bonjour ! Je suis votre assistant juridique. Posez-moi vos questions sur le droit tunisien.")
            return
            
        if query.lower().strip() in ["merci", "thank you", "شكرا"]:
            st.session_state.chat_history.append(HumanMessage(content=query))
            st.session_state.chat_history.append(AIMessage(content="Je vous en prie. N'hésitez pas si vous avez d'autres questions juridiques."))
            with st.chat_message("AI"):
                st.markdown("Je vous en prie. N'hésitez pas si vous avez d'autres questions juridiques.")
            return
        
        # Traitement des requêtes
        st.session_state.chat_history.append(HumanMessage(content=query))
        
        with st.chat_message("Human"):
            st.markdown(query)

        # Afficher le query rewriting seulement si c'est une requête juridique complexe
        if is_legal_query(query):
            rewritten_query = rewrite_legal_query(query, st.session_state.chat_history)
            if rewritten_query != query:
                with st.expander("Query Rewriting Process", expanded=False):
                    st.write("**Original Query:**")
                    st.info(query)
                    st.write("**Rewritten Query:**")
                    st.success(rewritten_query)
            
            question = rewritten_query
        else:
            question = query

        language = detect(query)
        language_name = {"ar": "Arabic", "fr": "French", "en": "English"}.get(language, "English")
        
        with st.chat_message("AI"):
            response = ""
            response_container = st.empty()
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    # Obtenir la réponse via RAG
                    llm = get_llm()
                    rag_response = get_response(llm, combined_index, question, st.session_state.chat_history)
                    
                    # Générer la réponse conversationnelle
                    for chunk in st.session_state.conversation.stream({
                        "question": rag_response,
                        "chat_history": st.session_state.chat_history,
                        "language": language_name
                    }):
                        response += chunk
                        response_container.markdown(response)
                    
                    st.session_state.chat_history.append(AIMessage(content=response))
                    st.session_state.memory.save_context({"question": query}, {"output": response})
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        response = "Désolé, le service est temporairement surchargé. Veuillez réessayer plus tard."
                        response_container.markdown(response)
                    time.sleep(retry_delay * (attempt + 1))

# Initialisation et routing
init_db()
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    choice = st.sidebar.selectbox("Choose an option", ["Login", "Register"])
    if choice == "Login":
        login()
    elif choice == "Register":
        register()
else:
    main()