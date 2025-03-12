import streamlit as st
import uuid
import boto3
import os
import tempfile
import re
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from unidecode import unidecode

# Configuration des logs
logging.basicConfig(level=logging.DEBUG)

# Set environment variables (replace these with your actual values)
os.environ['AWS_ACCESS_KEY_ID'] = 'AKIATWBJZ4G6IN7JILVW'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'c/gN8Vh9OKkFsJhZ3g+HrF8e4y0UkYdJwif0jCsJ'
os.environ['AWS_DEFAULT_REGION'] = 'eu-west-3'
os.environ['BUCKET_NAME'] = 'kb-pdf'
os.environ['API_TOKEN'] = 'IQoJb3JpZ2luX2VjEMj//////////wEaDGV1LWNlbnRyYWwtMSJGMEQCIFhvap2dp0k0Nh/9dWZW7nWZRNawgmkb8Vtv9nR1DsKaAiBhJ9kewsXYkSjEII318EP+bseL9f4pFplMwfkou9lsSSqeAwgREAAaDDI1MzQ5MDc0OTg4NCIMk2UDpqGHQcZiO0txKvsCa8EPMkJ0tmLEUy4Aww+0lVmQMmVNGw0+gR1DDLPWdwdtxVDq2/xHn5/MvntMnbu1w7blGKevzUV9WuDGRnongIjNmBzk20Z2vPRlCkTPaFHGkTB/XuEvD5NiEN189dm1TIhCAP1EYDsUv399yypEPxbQvgNXCzb9/U3nhCNnKcs0Xt0CYnOSbONbal0c3s9wsy9Ukrmx+Rt/+CnyPd+rNDna450MA93R0VGAeF/t0cqzn9s7BTSYU2ztsaZfM61/fCOdh0+lzGc4/pCnK/JbXLNV7aTzGLasGMCzOKoMM65mgudtsu9Fwj1ieAf5pic/maseBY/Lj4nP0Qxs1XhImg/aPq2b+vWANy3DFm+IBaGInOPAzbF53+o5vaoB1om0DW0HW/BLoDvJ7Lvnsb3h5FrNi41sKqOV6LuM1H5vwEQ2zFNJDm8N5PDL3Wq57F32B8gfG2gFuon1RI4Bl6qRDSN/b3lua9XilzlqPL5JN985F6eYgw9Zoe+0GTCmgKC+BjqnAbPt6XjQzHCV7BWHaeAgvFpzhd2LrdCGPYGroweyyrG7CH8tZpeI/i+b1auA128A59rxc7lM3ZBF2XfEpV/8soqtB4pCPt2Z7aUaatolhHI0q6vn8vPIFp6fD+fkhRUvtrkiVJ+FwH8mFPGJEMUHM0RfM1WoG9fBaN+Ryr/+4NGL3raGV9qiMKWdcalmh0cBr7OJMweHYO2GTB+wq4XjGay9S35Dw+U'  # Ajoutez votre token ici

# Log environment variables for debugging
# st.write(f"AWS_ACCESS_KEY_ID: {os.getenv('AWS_ACCESS_KEY_ID')}")
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

def normalize_filename(filename):
    """
    Normalise le nom du fichier en remplaçant les espaces et les caractères spéciaux par des underscores.
    """
    normalized_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
    return normalized_name

def split_by_articles(text):
    """
    Divise le texte en articles en utilisant un motif spécifique (ex: "Article 1", "Article 2").
    """
    articles = []
    current_article = ""
    for line in text.split("\n"):
        if line.strip().startswith("Article "):
            if current_article:
                articles.append(current_article.strip())
            current_article = line + "\n"
        else:
            current_article += line + "\n"
    if current_article:
        articles.append(current_article.strip())
    return articles

def process_pdf(file_path):
    """
    Charge un PDF et divise son contenu en articles.
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    full_text = "\n".join([page.page_content for page in pages])
    
    # Normaliser les caractères accentués
    full_text = unidecode(full_text)  # Convertit les accents en caractères non accentués
    articles = split_by_articles(full_text)
    return articles

def create_vector_store(text_chunks, document_name):
    """
    Crée un vecteur de stockage à partir des articles et l'enregistre.
    """
    # Normaliser le nom du document
    document_name = normalize_filename(document_name)

    #  afficher tous les chunks ou seulement un échantillon
    show_all_chunks = st.checkbox("Afficher tous les chunks de texte", value=False)

    if show_all_chunks:
        st.write("### Chunks de texte complets")
        for i, chunk in enumerate(text_chunks):   
            st.write(f"**Chunk {i+1}:**")
            st.write(chunk)   
            st.write("---")  
    else:
        st.write("### Exemple de chunks de texte")
        for i, chunk in enumerate(text_chunks[:3]):  
            st.write(f"**Chunk {i+1}:**")
            st.write(chunk[:500] + "...")   

    # Générer les embeddings
    st.write("### Exemple d'embeddings")
    sample_embedding = bedrock_embeddings.embed_query(text_chunks[0])   
    st.write(f"**Embedding du premier chunk:**")
    st.write(sample_embedding[:10])  # Afficher les 10 premières valeurs de l'embedding

    vectorstore_faiss = FAISS.from_texts(
        text_chunks,
        bedrock_embeddings
    )

    folder_path = tempfile.mkdtemp()  # Utilisez un dossier temporaire

    # Sauvegarder le vecteur de stockage
    vectorstore_faiss.save_local(folder_path=folder_path, index_name=document_name)

    # Vérifier que les fichiers ont été créés
    faiss_file = os.path.join(folder_path, f"{document_name}.faiss")
    pkl_file = os.path.join(folder_path, f"{document_name}.pkl")

    if not os.path.exists(faiss_file) or not os.path.exists(pkl_file):
        st.error(f"Erreur : Les fichiers FAISS ou PKL n'ont pas été créés pour {document_name}.")
        return False

    # Upload vers S3
    try:
        s3_client.upload_file(
            Filename=faiss_file,
            Bucket=BUCKET_NAME,
            Key=f"{document_name}.faiss"
        )

        s3_client.upload_file(
            Filename=pkl_file,
            Bucket=BUCKET_NAME,
            Key=f"{document_name}.pkl"
        )
    except Exception as e:
        st.error(f"Erreur lors de l'upload vers S3 : {e}")
        return False

    # Nettoyage des fichiers temporaires
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
        # Create a temporary directory to store uploaded files
        with tempfile.TemporaryDirectory() as tmp_dir:
            for uploaded_file in upload_folder:
                file_path = os.path.join(tmp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Process each PDF
                st.write(f"Processing {uploaded_file.name}")
                document_name = os.path.splitext(uploaded_file.name)[0]
                articles = process_pdf(file_path)
                st.write(f"Found {len(articles)} articles in {uploaded_file.name}")

                # Create a vector store for each PDF
                if create_vector_store(articles, document_name):
                    st.write(f"Vector store created for {uploaded_file.name}")
                else:
                    st.error(f"Failed to create vector store for {uploaded_file.name}")

            st.write("All vector stores created successfully")

if __name__ == "__main__":
    main()