import streamlit as st
import boto3
import os
import sqlite3
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

# Configuration des variables d'environnement
os.environ['AWS_ACCESS_KEY_ID'] = 'AKIATWBJZ4G6IN7JILVW'
os.environ['AWS_SECRET_ACCESS_KEY'] = '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'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
os.environ['BUCKET_NAME'] = 'kb-pdf'
 
# Initialisation du client Bedrock
bedrock_client = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

# Initialisation des embeddings
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock_client)

# Initialisation du client S3
s3_client = boto3.client('s3')
BUCKET_NAME = "kb-pdf"

folder_path = "local_data/" 
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Initialisation de la base de données SQLite
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS threads
                 (thread_id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, thread_name TEXT)''')
    conn.commit()
    conn.close()

# Ajouter un utilisateur à la base de données
def add_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        st.success("User registered successfully!")
        # Créer un thread par défaut pour l'utilisateur
        create_thread(username, "Default Thread")
    except sqlite3.IntegrityError:
        st.error("Username already exists. Please choose a different username.")
    finally:
        conn.close()

# Vérifier les informations de connexion
def authenticate(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    result = c.fetchone()
    conn.close()
    return result is not None

# Créer un nouveau thread pour un utilisateur
def create_thread(username, thread_name):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT INTO threads (username, thread_name) VALUES (?, ?)", (username, thread_name))
    thread_id = c.lastrowid
    conn.commit()
    conn.close()
    return thread_id

# Récupérer tous les threads d'un utilisateur
def get_threads(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT thread_id, thread_name FROM threads WHERE username = ?", (username,))
    threads = c.fetchall()
    conn.close()
    return threads



def get_response(llm, faiss_index, question):
    
    prompt_template = """

     ## 🔹 **Role**:
 You are a retriver. Your task is to **retrieve relevant legal articles** based on the user's question. You must provide **clear, precise, and professional responses** that directly address the user's legal query.

2. **Relevance**:
   - **Only return articles that directly address the user's question or problem.**
   - **Do NOT return irrelevant or approximate articles.**

3. **Response Format**:
   - Return only the **raw text** of the retrieved article(s), without any comments or explanations.
   - If **no relevant article** is found, respond clearly:
     > "No precise legal information found. Please provide more details about your situation."

 
 



Context: {context}  
Question: {question}  

Answer:

    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    answer = qa.invoke({"query": question})  # Utilisation de invoke avec la clé "query"
    return answer['result']

def get_llm():
    llm = ChatBedrock(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",   
        client=bedrock_client,
        temperature=0.3)
    return llm

def load_all_indices():
    # Lister tous les fichiers dans le bucket
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME)
    if 'Contents' not in response:
        st.error("No files found in the bucket.")
        return None

    # Télécharger tous les fichiers .faiss et .pkl
    for obj in response['Contents']:
        key = obj['Key']
        if key.endswith('.faiss') or key.endswith('.pkl'):
            local_path = os.path.join(folder_path, key)
            s3_client.download_file(
                Filename=local_path,
                Bucket=BUCKET_NAME,
                Key=key
            )

def create_conversation_chain(retriever):
    prompt_template = ChatPromptTemplate.from_messages([
        (""" ## 🔹 **Role**:
You are a **Tunisian lawyer**. Your task is to provide **clear, precise, and professional answers** based solely on the retrieved legal articles. You must help the user understand their legal situation, providing relevant explanations, clarifications, and examples when necessary.

## 🔹 **Guidelines**:
1. **Answer directly to the user's question**:
   - don't starting with phrases like "According to the context provided".
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
  
## 🔹 **Output Format**:
- **Formal, direct answers** with legal precision.
- Where necessary, **ask for clarification** if the question is ambiguous or requires more information.
- Remind the user to **consult a lawyer** for sensitive matters or when personal advice is needed.""" ),


 MessagesPlaceholder(variable_name="chat_history"),  # Placeholder pour l'historique de conversation
        ("human", "{question}"),
   ])
    
      # Initialiser la mémoire de conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Créer la chaîne de conversation
    chain = (
        RunnablePassthrough.assign(
            chat_history=lambda x: memory.load_memory_variables({})["chat_history"]
        )
        | prompt_template
        | get_llm()
        | StrOutputParser()
    )

    return chain, memory

# Interface d'inscription
def register():
    st.title("Register")
    username = st.text_input("Choose a username")
    password = st.text_input("Choose a password", type="password")
    if st.button("Register"):
        if username and password:
            add_user(username, password)
        else:
            st.error("Please fill in all fields.")

# Interface de connexion
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            st.session_state["username"] = username
            st.session_state["logged_in"] = True
            # Charger automatiquement le thread par défaut
            threads = get_threads(username)
            if threads:
                st.session_state["current_thread"] = threads[0][0]  # Charger le premier thread
                st.session_state["chat_history"] = []
            st.success("Logged in successfully!")
            st.rerun()
        else:
            st.error("Invalid username or password")

    

def main():
     
    st.header("Tunisian Legal Assistant with Memory")

    # Initialiser l'état de la session
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
        st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    load_all_indices()

    dir_list = os.listdir(folder_path)
    st.write("Files in /tmp/ directory")
    st.write(dir_list)

    # Charger tous les indices FAISS
    faiss_indices = []
    for file in dir_list:
        if file.endswith('.faiss'):
            index_name = file[:-6]  # Enlever l'extension .faiss
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

    if not faiss_indices:
        st.error("No valid FAISS indices found.")
        return

    st.write("Vector stores loaded successfully")

    # Fusionner tous les indices FAISS en un seul
    combined_index = faiss_indices[0]
    for index in faiss_indices[1:]:
        combined_index.merge_from(index)

    # Créer le retriever et la chaîne de conversation
    if st.session_state.retriever is None:
        st.session_state.retriever = combined_index.as_retriever(search_kwargs={"k": 3})
        st.session_state.conversation, st.session_state.memory = create_conversation_chain(st.session_state.retriever)

    # Main chat interface
    if st.session_state.conversation is None:
        st.info("Please upload and process documents to start asking questions.")
        return

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message("Human" if isinstance(message, HumanMessage) else "AI"):
            st.markdown(message.content)

   # Chat input
    query = st.chat_input("Ask a question about your documents")
    if query and st.session_state.logged_in and st.session_state.current_thread:
        # Ajouter le message de l'utilisateur à l'historique
        st.session_state.chat_history.append(HumanMessage(content=query))

        # Afficher le message de l'utilisateur
        with st.chat_message("Human"):
            st.markdown(query)

         # Detect language
        language = detect(query)
        language_name = {"ar": "Arabic", "fr": "French", "en": "English"}.get(language, "English")
         

         # Générer la réponse de l'IA
        with st.chat_message("AI"):
            response = ""
            response_container = st.empty()
            for chunk in st.session_state.conversation.stream({"question": query, "chat_history": st.session_state.chat_history, "language": language_name}):
                response += chunk
                response_container.markdown(response)
            st.session_state.chat_history.append(AIMessage(content=response))

        # Sauvegarder le contexte dans la mémoire
        st.session_state.memory.save_context({"question": query}, {"output": response})

# Initialiser la base de données
init_db()

# Interface de connexion ou inscription
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    choice = st.sidebar.selectbox("Choose an option", ["Login", "Register"])
    if choice == "Login":
        login()
    elif choice == "Register":
        register()
else:
    main()