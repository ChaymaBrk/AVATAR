import streamlit as st
import boto3
import os
import tempfile
import re
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from unidecode import unidecode
from dotenv import load_dotenv

# Configuration des logs
logging.basicConfig(level=logging.DEBUG)

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-west-3")
BUCKET_NAME = "kb-pdf"

s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION
)

bedrock_client = boto3.client(
    service_name='bedrock-runtime',
    region_name=AWS_DEFAULT_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock_client)

def normalize_filename(filename):
    """
    Normalise le nom du fichier en remplaçant les espaces et les caractères spéciaux par des underscores.
    """
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)

def extract_metadata(text, filename):
    """
    Extrait les métadonnées (code, titre, chapitre, section) d'un texte juridique structuré.
    """
    metadata = {
        "law_code": filename,  # Directly assign filename as law_code
        "title": "",
        "chapter": "",
        "section": ""
    }

    # Recherche de Titre
    match = re.search(r"Titre\s+([\w\s]+)", text, re.IGNORECASE)
    if match:
        metadata["title"] = match.group(1).strip()

    # Recherche de Chapitre
    match = re.search(r"Chapitre\s+([\w\s]+)", text, re.IGNORECASE)
    if match:
        metadata["chapter"] = match.group(1).strip()

    # Recherche de Section
    match = re.search(r"Section\s+([\w\s]+)", text, re.IGNORECASE)
    if match:
        metadata["section"] = match.group(1).strip()
    
    print(metadata)

    return metadata

def split_by_articles(text, filename):
    """
    Divise le texte en articles en utilisant un motif spécifique (ex: "Article 1", "Article 2").
    """
    articles = []
    metadata_list = []
    current_article = ""
    current_metadata = {}

    for line in text.split("\n"):
        if line.strip().startswith("Article "):
            if current_article:
                articles.append(current_article.strip())
                metadata_list.append(current_metadata)
            
            current_article = line + "\n"
            current_metadata = extract_metadata(line, filename)  # Pass the filename here
        else:
            current_article += line + "\n"

    if current_article:
        articles.append(current_article.strip())
        metadata_list.append(current_metadata)

    return articles, metadata_list


def process_pdf(file_path):
    """
    Charge un PDF et divise son contenu en articles avec extraction des métadonnées.
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    full_text = "\n".join([page.page_content for page in pages])
    
    # Normaliser les caractères accentués
    full_text = unidecode(full_text)  # Convertit les accents en caractères non accentués

    filename = os.path.basename(file_path)  # Get the file name

    # Découpage en articles et extraction des métadonnées
    articles, metadata_list = split_by_articles(full_text, filename)  # Pass the filename
    
    return articles, metadata_list


def create_vector_store(text_chunks, metadata_list, document_name):
    """
    Crée un vecteur de stockage avec des métadonnées et l'enregistre.
    """
    document_name = normalize_filename(document_name)

    # Créer des objets Document avec le texte et la métadonnée associée
    docs = [
        Document(page_content=chunk, metadata=metadata)
        for chunk, metadata in zip(text_chunks, metadata_list)
    ]

    # Créer un index vectoriel avec FAISS
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)

    folder_path = tempfile.mkdtemp()

    # Sauvegarder l'index localement
    vectorstore_faiss.save_local(folder_path=folder_path, index_name=document_name)

    # Vérifier que les fichiers ont été créés
    faiss_file = os.path.join(folder_path, f"{document_name}.faiss")
    pkl_file = os.path.join(folder_path, f"{document_name}.pkl")

    if not os.path.exists(faiss_file) or not os.path.exists(pkl_file):
        st.error(f"Erreur : Les fichiers FAISS ou PKL n'ont pas été créés pour {document_name}.")
        return False

    # Upload vers S3
    try:
        s3_client.upload_file(Filename=faiss_file, Bucket=BUCKET_NAME, Key=f"{document_name}.faiss")
        s3_client.upload_file(Filename=pkl_file, Bucket=BUCKET_NAME, Key=f"{document_name}.pkl")
    except Exception as e:
        st.error(f"Erreur lors de l'upload vers S3 : {e}")
        return False

    # Nettoyage des fichiers tempor,aires
    try:
        os.remove(faiss_file)
        os.remove(pkl_file)
        os.rmdir(folder_path)
    except Exception as e:
        st.error(f"Erreur lors du nettoyage des fichiers temporaires : {e}")

    return True

def main():
    st.write("This is Admin Site for PDF chat bot application")
    upload_folder = st.file_uploader("Upload a folder containing PDF files", type="pdf", accept_multiple_files=True)

    if upload_folder:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for uploaded_file in upload_folder:
                file_path = os.path.join(tmp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Process PDF and extract metadata
                st.write(f"Processing {uploaded_file.name}")
                document_name = os.path.splitext(uploaded_file.name)[0]
                articles, metadata_list = process_pdf(file_path)
                st.write(f"Found {len(articles)} articles in {uploaded_file.name}")

                # Create vector store with metadata
                if create_vector_store(articles, metadata_list, document_name):
                    st.write(f"Vector store created for {uploaded_file.name}")
                else:
                    st.error(f"Failed to create vector store for {uploaded_file.name}")

            st.write("All vector stores created successfully")

if __name__ == "__main__":
    main()
