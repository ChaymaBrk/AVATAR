import streamlit as st
import uuid
import boto3
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS

# Set environment variables (replace these with your actual values)
os.environ['AWS_ACCESS_KEY_ID'] = 'AKIATWBJZ4G6IN7JILVW'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'c/gN8Vh9OKkFsJhZ3g+HrF8e4y0UkYdJwif0jCsJ'
os.environ['AWS_DEFAULT_REGION'] = 'eu-west-3'
os.environ['BUCKET_NAME'] = 'kb-pdf'
os.environ['API_TOKEN'] = 'IQoJb3JpZ2luX2VjEMj//////////wEaDGV1LWNlbnRyYWwtMSJGMEQCIFhvap2dp0k0Nh/9dWZW7nWZRNawgmkb8Vtv9nR1DsKaAiBhJ9kewsXYkSjEII318EP+bseL9f4pFplMwfkou9lsSSqeAwgREAAaDDI1MzQ5MDc0OTg4NCIMk2UDpqGHQcZiO0txKvsCa8EPMkJ0tmLEUy4Aww+0lVmQMmVNGw0+gR1DDLPWdwdtxVDq2/xHn5/MvntMnbu1w7blGKevzUV9WuDGRnongIjNmBzk20Z2vPRlCkTPaFHGkTB/XuEvD5NiEN189dm1TIhCAP1EYDsUv399yypEPxbQvgNXCzb9/U3nhCNnKcs0Xt0CYnOSbONbal0c3s9wsy9Ukrmx+Rt/+CnyPd+rNDna450MA93R0VGAeF/t0cqzn9s7BTSYU2ztsaZfM61/fCOdh0+lzGc4/pCnK/JbXLNV7aTzGLasGMCzOKoMM65mgudtsu9Fwj1ieAf5pic/maseBY/Lj4nP0Qxs1XhImg/aPq2b+vWANy3DFm+IBaGInOPAzbF53+o5vaoB1om0DW0HW/BLoDvJ7Lvnsb3h5FrNi41sKqOV6LuM1H5vwEQ2zFNJDm8N5PDL3Wq57F32B8gfG2gFuon1RI4Bl6qRDSN/b3lua9XilzlqPL5JN985F6eYgw9Zoe+0GTCmgKC+BjqnAbPt6XjQzHCV7BWHaeAgvFpzhd2LrdCGPYGroweyyrG7CH8tZpeI/i+b1auA128A59rxc7lM3ZBF2XfEpV/8soqtB4pCPt2Z7aUaatolhHI0q6vn8vPIFp6fD+fkhRUvtrkiVJ+FwH8mFPGJEMUHM0RfM1WoG9fBaN+Ryr/+4NGL3raGV9qiMKWdicalmh0cBr7OJMweHYO2GTB+wq4XjGay9S35Dw+U'  # Ajoutez votre token ici

# Log environment variables for debugging
#st.write(f"AWS_ACCESS_KEY_ID: {os.getenv('AWS_ACCESS_KEY_ID')}")
# st.write(f"AWS_SECRET_ACCESS_KEY: {os.getenv('AWS_SECRET_ACCESS_KEY')}")
# st.write(f"BUCKET_NAME: {os.getenv('BUCKET_NAME')}")
# st.write(f"API_TOKEN: {os.getenv('API_TOKEN')}")  # Affichez le token pour le débogage

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "eu-west-3")
)

BUCKET_NAME = "kb-pdf"

bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='eu-west-3', aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock_client)

def split_text(pages, chunk_size=1000, overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    docs = text_splitter.split_documents(pages)
    return docs

def create_vector_store(text_chunks, req_id):
    vectorstore_faiss = FAISS.from_documents(
        text_chunks,
        bedrock_embeddings
    )

    file_name = f"{req_id}.bin"

    folder_path = tempfile.mkdtemp()  # Utilisez un dossier temporaire

    vectorstore_faiss.save_local(folder_path=folder_path, index_name=file_name)

    s3_client.upload_file(
        Filename=os.path.join(folder_path, f"{file_name}.faiss"),
        Bucket=BUCKET_NAME,
        Key="my_faiss.faiss"
    )

    s3_client.upload_file(
        Filename=os.path.join(folder_path, f"{file_name}.pkl"),
        Bucket=BUCKET_NAME,
        Key="my_faiss.pkl"
    )

    # Nettoyage des fichiers temporaires
    os.remove(os.path.join(folder_path, f"{file_name}.faiss"))
    os.remove(os.path.join(folder_path, f"{file_name}.pkl"))
    os.rmdir(folder_path)

    return True

def get_uuid():
    return str(uuid.uuid4())

def main():
    st.write("This is Admin Site for PDF chat bot application")
    upload_file = st.file_uploader("Upload a PDF file", type="pdf")

    if upload_file is not None:
        req_id = get_uuid()
        st.write(f"File Request ID: {req_id}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(upload_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load_and_split()

        st.write(f"Total no. of pages : {len(pages)}")

        split_document = split_text(pages, 1000, 200)
        st.write(f"Total no. of chunks : {len(split_document)}")

        st.write("Embedding the document chunks")
        response = create_vector_store(split_document, req_id)

        if response:
            st.write("Vector store created successfully")
        else:
            st.write("Vector store creation failed")

        # Nettoyage du fichier temporaire
        os.remove(tmp_file_path)

if __name__ == "__main__":
    main()