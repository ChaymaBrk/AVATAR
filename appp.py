import streamlit as st
import boto3
import os
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import HumanMessage, AIMessage  # Pour gérer les messages de chat
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configuration des variables d'environnement
os.environ['AWS_ACCESS_KEY_ID'] = 'AKIATWBJZ4G6IN7JILVW'
os.environ['AWS_SECRET_ACCESS_KEY'] = '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'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
os.environ['BUCKET_NAME'] = 'kb-pdf'
os.environ['API_TOKEN'] = 'IQoJb3JpZ2luX2VjEMj//////////wEaDGV1LWNlbnRyYWwtMSJGMEQCIFhvap2dp0k0Nh/9dWZW7nWZRNawgmkb8Vtv9nR1DsKaAiBhJ9kewsXYkSjEII318EP+bseL9f4pFplMwfkou9lsSSqeAwgREAAaDDI1MzQ5MDc0OTg4NCIMk2UDpqGHQcZiO0txKvsCa8EPMkJ0tmLEUy4Aww+0lVmQMmVNGw0+gR1DDLPWdwdtxVDq2/xHn5/MvntMnbu1w7blGKevzUV9WuDGRnongIjNmBzk20Z2vPRlCkTPaFHGkTB/XuEvD5NiEN189dm1TIhCAP1EYDsUv399yypEPxbQvgNXCzb9/U3nhCNnKcs0Xt0CYnOSbONbal0c3s9wsy9Ukrmx+Rt/+CnyPd+rNDna450MA93R0VGAeF/t0cqzn9s7BTSYU2ztsaZfM61/fCOdh0+lzGc4/pCnK/JbXLNV7aTzGLasGMCzOKoMM65mgudtsu9Fwj1ieAf5pic/maseBY/Lj4nP0Qxs1XhImg/aPq2b+vWANy3DFm+IBaGInOPAzbF53+o5vaoB1om0DW0HW/BLoDvJ7Lvnsb3h5FrNi41sKqOV6LuM1H5vwEQ2zFNJDm8N5PDL3Wq57F32B8gfG2gFuon1RI4Bl6qRDSN/b3lua9XilzlqPL5JN985F6eYgw9Zoe+0GTCmgKC+BjqnAbPt6XjQzHCV7BWHaeAgvFpzhd2LrdCGPYGroweyyrG7CH8tZpeI/i+b1auA128A59rxc7lM3ZBF2XfEpV/8soqtB4pCPt2Z7aUaatolhHI0q6vn8vPIFp6fD+fkhRUvtrkiVJ+FwH8mFPGJEMUHM0RfM1WoG9fBaN+Ryr/+4NGL3raGV9qiMKWdcalmh0cBr7OJMweHYO2GTB+wq4XjGay9S35Dw+U'

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

folder_path = "/tmp/"

def get_response(llm, faiss_index, question):
    prompt_template = """

    You are a Tunisian lawyer. Your role is to provide accurate, concise, and well-structured responses based on the retrieved legal context. Follow these guidelines:

1. **Contextual Understanding**:
   - Use the provided legal context to answer the question. If the context is insufficient or irrelevant, ask for more clarification.

3. **Legal Precision**:
   - Ensure that your answers are legally accurate and avoid ambiguous language.
   - If the question involves interpretation, clarify whether your response is based on the provided context or general legal principles.

4. **Ethical Responsibility**:
   - If the question involves sensitive legal matters (e.g., personal legal advice), remind the user to consult a qualified legal professional.

5. **Formatting**:
   - Use clear and professional language. Structure your response with headings, bullet points, or numbered lists when appropriate to improve readability.

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
        client=bedrock_client)
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
    prompt_template = """

    You are a Tunisian lawyer. Your role is to provide accurate, concise, and well-structured responses based on the retrieved legal context. Follow these guidelines:

1. **Contextual Understanding**:
   - Use the provided legal context to answer the question. If the context is insufficient or irrelevant, ask for more clarification.

3. **Legal Precision**:
   - Ensure that your answers are legally accurate and avoid ambiguous language.
   - If the question involves interpretation, clarify whether your response is based on the provided context or general legal principles.

4. **Ethical Responsibility**:
   - If the question involves sensitive legal matters (e.g., personal legal advice), remind the user to consult a qualified legal professional.

5. **Formatting**:
   - Use clear and professional language. Structure your response with headings, bullet points, or numbered lists when appropriate to improve readability.

Context: {context}  
Question: {question}  

Answer:

    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = get_llm()

    def format_context(inputs):
        docs = retriever.invoke(inputs["question"])
        return {
            **inputs,
            "context": "\n\n".join([doc.page_content for doc in docs])
        }

    chain = (
        RunnablePassthrough() 
        | format_context
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

def main():
    st.header("This is client side PDF Chat bot")

    # Initialiser l'état de la session
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

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
        st.session_state.conversation = create_conversation_chain(st.session_state.retriever)

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
    if query:
        st.session_state.chat_history.append(HumanMessage(content=query))
        
        # Display user message
        with st.chat_message("Human"):
            st.markdown(query)

        # Generate and display AI response
        with st.chat_message("AI"):
            response = ""
            response_container = st.empty()
            
            for chunk in st.session_state.conversation.stream({"question": query}):
                response += chunk
                response_container.markdown(response)
            
            # Add AI response to chat history
            st.session_state.chat_history.append(AIMessage(content=response))

if __name__ == "__main__":
    main()