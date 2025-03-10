import streamlit as st
import boto3
import os

from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings, ChatBedrock   
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

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

    Human: Use the following pieces of context to provide a concise answer to the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}

    Assistant:
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    answer = qa.invoke({"query": question})  # Utilisation de invoke au lieu de __call__
    return answer['result']

def get_llm():
    llm = ChatBedrock(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",   
        client=bedrock_client)
    return llm

def load_index():
    s3_client.download_file(
        Filename=f"{folder_path}my_faiss.faiss",
        Bucket=BUCKET_NAME,
        Key="my_faiss.faiss"
    )

    s3_client.download_file(
        Filename=f"{folder_path}my_faiss.pkl",
        Bucket=BUCKET_NAME,
        Key="my_faiss.pkl"
    )

def main():
    st.header("This is client side PDF Chat bot")

    load_index()

    dir_list = os.listdir("/tmp/")
    st.write("Files in /tmp/ directory")
    st.write(dir_list)

    faiss_index = FAISS.load_local(
        index_name="my_faiss",
        folder_path=folder_path,
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True
    )

    st.write("Vector store loaded successfully")
    question = st.text_input("Enter your question")
    if st.button("Get Response"):
        with st.spinner("Getting response..."):
            llm = get_llm()
            st.write(get_response(llm, faiss_index, question))
            st.success("Done!")

if __name__ == "__main__":
    main()