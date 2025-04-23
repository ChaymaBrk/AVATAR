import firebase_admin
from firebase_admin import credentials, auth, firestore, exceptions
from datetime import datetime
import requests
import logging
import streamlit as st

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation Firebase
def initialize_firebase():
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate("firebase_creds.json")
            firebase_admin.initialize_app(cred)
            logger.info("Firebase initialized successfully")
        
        db = firestore.client()
        # Test de connexion
        test_ref = db.collection("test_connection").document("test")
        test_ref.set({"test": True})
        return db
    except exceptions.FirebaseError as e:
        logger.error(f"Firebase initialization error: {e}")
        st.error("""
            Erreur d'initialisation Firebase. Vérifiez que:
            1. Le fichier de credentials est valide
            2. Firestore est activé dans la console Firebase
            3. Les règles de sécurité permettent l'accès
        """)
        st.stop()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        st.stop()

db = initialize_firebase()

# Authentification
def firebase_sign_in(email, password):
    try:
        email = email.strip().lower()
        response = requests.post(
            "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword",
            params={"key": "AIzaSyAc_Tnh_wbPRSF7qE49xFGEx0d1CdpknWs"},
            json={"email": email, "password": password, "returnSecureToken": True}
        )
        
        if response.status_code == 200:
            data = response.json()
            uid = data.get("localId")
            user = auth.get_user(uid)
            
            return {
                'uid': user.uid,
                'email': user.email,
                'username': user.display_name or email.split('@')[0]
            }
        else:
            error = response.json().get('error', {}).get('message', 'Unknown error')
            raise Exception(f"Firebase auth error: {error}")
    
    except Exception as e:
        logger.error(f"Sign in error: {e}")
        raise

def firebase_sign_up(email, password, username):
    try:
        email = email.strip().lower()
        user = auth.create_user(
            email=email,
            password=password,
            display_name=username
        )
        
        # Création du document utilisateur
        user_ref = db.collection('users').document(user.uid)
        user_ref.set({
            'email': email,
            'username': username,
            'created_at': datetime.now()
        })
        
        # Création d'un thread par défaut
        user_ref.collection('threads').add({
            'thread_name': 'Default Thread',
            'created_at': datetime.now(),
            'messages': []
        })
        
        return user.uid
    except auth.EmailAlreadyExistsError:
        raise Exception("Email already exists")
    except Exception as e:
        logger.error(f"Sign up error: {e}")
        raise

# Gestion des conversations
def save_chat_history(uid, thread_id, question, answer):
    try:
        thread_ref = db.collection('users').document(uid)\
                      .collection('threads').document(thread_id)
        
        thread_data = thread_ref.get().to_dict()
        messages = thread_data.get('messages', [])
        
        messages.append({
            'content': question,
            'response': answer,
            'timestamp': datetime.now().isoformat()
        })
        
        thread_ref.update({'messages': messages})
        return True
    except Exception as e:
        logger.error(f"Error saving conversation: {e}")
        return False

def get_user_history(uid, thread_id, limit=50):
    try:
        thread_ref = db.collection('users').document(uid)\
                      .collection('threads').document(thread_id)
        thread_data = thread_ref.get().to_dict()
        
        if not thread_data:
            return []
            
        messages = thread_data.get('messages', [])
        # Trier par timestamp et limiter les résultats
        sorted_messages = sorted(messages, key=lambda x: x['timestamp'], reverse=True)[:limit]
        return [(msg['content'], msg['response'], msg['timestamp']) for msg in sorted_messages]
    except Exception as e:
        logger.error(f"Error getting user history: {e}")
        return []

def clear_user_history(uid, thread_id):
    try:
        thread_ref = db.collection('users').document(uid)\
                      .collection('threads').document(thread_id)
        thread_ref.update({'messages': []})
        return True
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        return False

def get_user_threads(uid):
    try:
        threads_ref = db.collection('users').document(uid).collection('threads')
        return {doc.id: doc.to_dict() for doc in threads_ref.stream()}
    except Exception as e:
        logger.error(f"Error getting user threads: {e}")
        return {}