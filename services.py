import os
import boto3
import streamlit as st
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import time
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from config import AWS_CONFIG, MODEL_CONFIG, APP_CONFIG
from utils import detect_language

# Décorateur pour mesurer le temps d'exécution
def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result
    return wrapper

@timeit
def init_bedrock_client():
    """Initialize AWS Bedrock client"""
    return boto3.client(
        service_name='bedrock-runtime',
        region_name=AWS_CONFIG['AWS_DEFAULT_REGION'],
        aws_access_key_id=AWS_CONFIG['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=AWS_CONFIG['AWS_SECRET_ACCESS_KEY']
    )

@timeit
def init_embeddings():
    """Initialize embeddings model"""
    return BedrockEmbeddings(
        model_id=MODEL_CONFIG['EMBEDDING_MODEL'], 
        client=init_bedrock_client()
    )

@timeit
def init_s3_client():
    """Initialize S3 client"""
    return boto3.client('s3')

@timeit
def load_all_indices(s3_client):
    """Load all FAISS indices from S3"""
    if not os.path.exists(APP_CONFIG['FOLDER_PATH']):
        os.makedirs(APP_CONFIG['FOLDER_PATH'])
    
    response = s3_client.list_objects_v2(Bucket=AWS_CONFIG['BUCKET_NAME'])
    if 'Contents' not in response:
        st.error("No files found in the bucket.")
        return None

    s3_files = {obj['Key']: obj['LastModified'].replace(tzinfo=None) for obj in response['Contents'] 
                if obj['Key'].endswith(('.faiss', '.pkl'))}

    def download_file(key):
        local_path = os.path.join(APP_CONFIG['FOLDER_PATH'], key)
        
        if not os.path.exists(local_path) or \
           (key in s3_files and 
            datetime.fromtimestamp(os.path.getmtime(local_path)).replace(tzinfo=None) < s3_files[key]):
            
            s3_client.download_file(
                Filename=local_path,
                Bucket=AWS_CONFIG['BUCKET_NAME'],
                Key=key
            )

    with ThreadPoolExecutor(max_workers=APP_CONFIG['MAX_WORKERS']) as executor:
        executor.map(download_file, s3_files.keys())

@timeit
def load_faiss_indices():
    """Load FAISS indices from local folder"""
    dir_list = os.listdir(APP_CONFIG['FOLDER_PATH'])
    faiss_indices = []
    for file in dir_list:
        if file.endswith('.faiss'):
            index_name = file[:-6]
            try:
                faiss_index = FAISS.load_local(
                    index_name=index_name,
                    folder_path=APP_CONFIG['FOLDER_PATH'],
                    embeddings=init_embeddings(),
                    allow_dangerous_deserialization=True
                )
                faiss_indices.append(faiss_index)
            except Exception as e:
                st.error(f"Failed to load index {index_name}: {e}")
    return faiss_indices

@timeit
def merge_indices(faiss_indices):
    """Merge multiple FAISS indices into one"""
    if not faiss_indices:
        st.error("No valid FAISS indices found.")
        return None

    combined_index = faiss_indices[0]
    for index in faiss_indices[1:]:
        combined_index.merge_from(index)
    return combined_index

@timeit
def get_llm(bedrock_client):
    """Initialize the LLM model"""
    return ChatBedrock(
        model_id=MODEL_CONFIG['LLM_MODEL'],
        client=bedrock_client,
        temperature=MODEL_CONFIG['TEMPERATURE']
    )

@timeit
def get_response(llm, faiss_index, question):
    """Get response from the retrieval system"""
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

    answer = qa.invoke({"query": question})
    return answer['result']

@timeit
def create_conversation_chain(retriever, bedrock_client):
    """Create conversation chain with memory"""
    prompt_template = ChatPromptTemplate.from_messages([
        (""" ## 🔹 **Role**:
You are a **Tunisian Legal Agent**. Your task is to provide **clear, precise, and professional answers** based solely on the retrieved legal articles. You must help the user understand their legal situation, providing relevant explanations, clarifications, and examples when necessary.

## 🔹 **Guidelines**:
1. **Answer directly to the user's question**:
   - don't starting with phrases like "According to the context provided".
   - Provide a **concise, direct answer** for straightforward or factual questions.
   - For broader or more complex questions, offer a **detailed, well-structured response** with examples, legal references, or relevant case law where applicable.
   
2. **Level of Detail**:
   - **If the question is broad, complex, or requires interpretation**, give a **detailed explanation**.
   - **If the question is straightforward**, give a **concise, direct answer**.

3. **Legal Precision**:
   - Ensure all responses are **legally accurate**, clear, and free from ambiguity.
   
4. **Ethical Responsibility**:
   - If the question concerns **personal or sensitive legal matters**, always remind the user to **consult a qualified lawyer**.

5. **Formatting**:
   - **Use clear and professional language**. Avoid colloquial or casual speech.
   - **Structure your responses for readability**.
   
6. **Language**:
   - **Always reply in the same language used by the user: {language}**, and **formally**
   - The response must be **formal**, professional, and precise.

7. **Clarification**:
   - If the provided legal articles do not fully answer the question, **politely ask the user for further details** or clarification.
  
## 🔹 **Output Format**:
- **Formal, direct answers** with legal precision.
- Where necessary, **ask for clarification** if the question is ambiguous.
- Remind the user to **consult a lawyer** for sensitive matters."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = (
        RunnablePassthrough.assign(
            chat_history=lambda x: memory.load_memory_variables({})["chat_history"]
        )
        | prompt_template
        | get_llm(bedrock_client)
        | StrOutputParser()
    )

    return chain, memory

@timeit
def create_reviewer_chain(bedrock_client):
    """Create the reviewer chain that checks legal responses"""
    reviewer_prompt = ChatPromptTemplate.from_messages([
        ("""## 🔹 **Role**: 
You are a **Senior Legal Reviewer**. Your task is to review legal responses generated by other agents to ensure they meet the highest standards of legal accuracy, clarity, and professionalism.

## 🔹 **Review Guidelines**:
1. **Legal Accuracy**:
   - Verify that all legal references are correct and up-to-date
   - Ensure interpretations of law are sound and defensible
   - Flag any potential misstatements or oversimplifications

2. **Clarity & Readability**:
   - Ensure the response is well-structured and easy to understand
   - Check for ambiguous language or confusing phrasing
   - Verify technical terms are properly explained

3. **Completeness**:
   - Ensure the response fully addresses the user's question
   - Check if important aspects or edge cases are missing
   - Verify that disclaimers are included where needed

4. **Professionalism**:
   - Ensure tone is consistently professional
   - Check for inappropriate language or colloquialisms
   - Verify formatting is clean and consistent

5. **Ethical Compliance**:
   - Ensure appropriate disclaimers about consulting a lawyer are present
   - Verify no specific legal advice is given for personal situations
   - Check that confidentiality is maintained

## 🔹 **Output Format**:
Provide your reviewed response in this format:

**Reviewed Response**:
[Your improved version of the response]

**Review Notes**:
- [List of specific changes made]
- [Any important considerations for the user]
- [Additional recommendations if applicable]"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Original response to review:\n\n{response}"),
    ])
    
    reviewer_chain = (
        RunnablePassthrough.assign(
            chat_history=lambda x: x["chat_history"]
        )
        | reviewer_prompt
        | get_llm(bedrock_client)
        | StrOutputParser()
    )
    
    return reviewer_chain