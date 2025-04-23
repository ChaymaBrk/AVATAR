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

# Configuration AWS (à remplacer par vos vraies valeurs)
os.environ['AWS_ACCESS_KEY_ID'] = 'AKIATWBJZ4G6IN7JILVW'
os.environ['AWS_SECRET_ACCESS_KEY'] = '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'
os.environ['AWS_DEFAULT_REGION'] = 'eu-west-3'
os.environ['BUCKET_NAME'] = 'kb-pdf'
os.environ['API_TOKEN'] = 'IQoJb3JpZ2luX2VjEMj//////////wEaDGV1LWNlbnRyYWwtMSJGMEQCIFhvap2dp0k0Nh/9dWZW7nWZRNawgmkb8Vtv9nR1DsKaAiBhJ9kewsXYkSjEII318EP+bseL9f4pFplMwfkou9lsSSqeAwgREAAaDDI1MzQ5MDc0OTg4NCIMk2UDpqGHQcZiO0txKvsCa8EPMkJ0tmLEUy4Aww+0lVmQMmVNGw0+gR1DDLPWdwdtxVDq2/xHn5/MvntMnbu1w7blGKevzUV9WuDGRnongIjNmBzk20Z2vPRlCkTPaFHGkTB/XuEvD5NiEN189dm1TIhCAP1EYDsUv399yypEPxbQvgNXCzb9/U3nhCNnKcs0Xt0CYnOSbONbal0c3s9wsy9Ukrmx+Rt/+CnyPd+rNDna450MA93R0VGAeF/t0cqzn9s7BTSYU2ztsaZfM61/fCOdh0+lzGc4/pCnK/JbXLNV7aTzGLasGMCzOKoMM65mgudtsu9Fwj1ieAf5pic/maseBY/Lj4nP0Qxs1XhImg/aPq2b+vWANy3DFm+IBaGInOPAzbF53+o5vaoB1om0DW0HW/BLoDvJ7Lvnsb3h5FrNi41sKqOV6LuM1H5vwEQ2zFNJDm8N5PDL3Wq57F32B8gfG2gFuon1RI4Bl6qRDSN/b3lua9XilzlqPL5JN985F6eYgw9Zoe+0GTCmgKC+BjqnAbPt6XjQzHCV7BWHaeAgvFpzhd2LrdCGPYGroweyyrG7CH8tZpeI/i+b1auA128A59rxc7lM3ZBF2XfEpV/8soqtB4pCPt2Z7aUaatolhHI0q6vn8vPIFp6fD+fkhRUvtrkiVJ+FwH8mFPGJEMUHM0RfM1WoG9fBaN+Ryr/+4NGL3raGV9qiMKWdcalmh0cBr7OJMweHYO2GTB+wq4XjGay9S35Dw+U'

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "eu-west-3")
)

BUCKET_NAME = "kb-pdf"
MAX_CHUNK_LENGTH = 10000  # Longueur maximale autorisée pour un chunk

bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='eu-west-3', 
                             aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                             aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock_client)

def normalize_filename(filename):
    """Normalise le nom du fichier."""
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)

def split_by_articles(text):
    """Divise le texte en articles."""
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
    """Charge et traite un PDF."""
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    full_text = "\n".join([page.page_content for page in pages])
    full_text = unidecode(full_text)
    return split_by_articles(full_text)

def check_long_chunks(text_chunks, source_name):
    """Identifie et affiche les chunks trop longs."""
    long_chunks = []
    for i, chunk in enumerate(text_chunks):
        if len(chunk) > MAX_CHUNK_LENGTH:
            long_chunks.append({
                "index": i+1,
                "length": len(chunk),
                "start": chunk[:100] + "...",
                "source": source_name
            })
    
    if long_chunks:
        st.warning("⚠️ Certains chunks sont trop longs et pourraient causer des problèmes:")
        for chunk in long_chunks:
            st.write(f"**Chunk {chunk['index']}** (longueur: {chunk['length']} caractères)")
            st.write(f"Source: {chunk['source']}")
            st.write(f"Extrait: {chunk['start']}")
            st.write("---")
        
        if st.button("Afficher les options de correction"):
            st.session_state.show_correction = True
    
    return long_chunks

def create_vector_store(text_chunks, document_name):
    """Crée et enregistre le vector store."""
    document_name = normalize_filename(document_name)
    
    # Vérification des chunks trop longs
    long_chunks = check_long_chunks(text_chunks, document_name)
    
    if long_chunks and st.session_state.get('show_correction', False):
        st.write("### Options de correction")
        selected_chunk = st.selectbox(
            "Sélectionnez un chunk à corriger",
            options=[f"Chunk {c['index']}" for c in long_chunks]
        )
        
        chunk_index = int(selected_chunk.split()[1]) - 1
        current_content = st.text_area(
            "Contenu actuel",
            value=text_chunks[chunk_index],
            height=300
        )
        
        if st.button("Appliquer les modifications"):
            text_chunks[chunk_index] = current_content
            st.success("Modification appliquée!")
            st.session_state.show_correction = False
            st.experimental_rerun()

    # Affichage des chunks
    show_all_chunks = st.checkbox("Afficher tous les chunks", value=False)
    if show_all_chunks:
        for i, chunk in enumerate(text_chunks):
            st.write(f"**Chunk {i+1}:**")
            st.write(chunk)
            st.write("---")
    else:
        for i, chunk in enumerate(text_chunks[:3]):
            st.write(f"**Chunk {i+1}:**")
            st.write(chunk[:500] + "...")

    # Création du vector store
    with st.spinner("Création du vector store..."):
        try:
            vectorstore = FAISS.from_texts(text_chunks, bedrock_embeddings)
            folder_path = tempfile.mkdtemp()
            
            vectorstore.save_local(folder_path=folder_path, index_name=document_name)
            
            # Upload vers S3
            for ext in ['faiss', 'pkl']:
                file_path = os.path.join(folder_path, f"{document_name}.{ext}")
                s3_client.upload_file(file_path, BUCKET_NAME, f"{document_name}.{ext}")
            
            return True
        except Exception as e:
            st.error(f"Erreur: {str(e)}")
            return False

def main():
    st.title("Admin Site for PDF Chat Bot")
    st.write("Téléchargez des fichiers PDF pour créer des vector stores")
    
    uploaded_files = st.file_uploader(
        "Sélectionnez des fichiers PDF",
        type="pdf",
        accept_multiple_files=True
    )
    
    if uploaded_files:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for uploaded_file in uploaded_files:
                st.write(f"### Traitement de: {uploaded_file.name}")
                
                # Sauvegarde temporaire du fichier
                file_path = os.path.join(tmp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Traitement du PDF
                try:
                    articles = process_pdf(file_path)
                    st.success(f"{len(articles)} articles trouvés")
                    
                    # Création du vector store
                    if create_vector_store(articles, os.path.splitext(uploaded_file.name)[0]):
                        st.success("Vector store créé avec succès!")
                    else:
                        st.error("Échec de la création du vector store")
                
                except Exception as e:
                    st.error(f"Erreur lors du traitement: {str(e)}")

if __name__ == "__main__":
    if 'show_correction' not in st.session_state:
        st.session_state.show_correction = False
    main()