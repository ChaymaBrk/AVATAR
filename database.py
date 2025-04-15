import sqlite3
import streamlit as st
from functools import wraps
import time
import json
from datetime import datetime

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
def init_db():
    """Initialize the SQLite database"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Table utilisateurs
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')
    
    # Table historique
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  timestamp DATETIME,
                  question TEXT,
                  answer TEXT,
                  FOREIGN KEY(username) REFERENCES users(username))''')
    
    # Index pour améliorer les performances
    c.execute('''CREATE INDEX IF NOT EXISTS idx_username ON chat_history(username)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_timestamp ON chat_history(timestamp)''')
    
    conn.commit()
    conn.close()

@timeit
def add_user(username, password):
    """Add a new user to the database"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        st.success("User registered successfully!")
    except sqlite3.IntegrityError:
        st.error("Username already exists. Please choose a different username.")
    finally:
        conn.close()

@timeit
def authenticate(username, password):
    """Authenticate a user"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    result = c.fetchone()
    conn.close()
    return result is not None

@timeit
def save_chat_history(username, question, answer):
    """Save conversation history"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO chat_history (username, timestamp, question, answer) VALUES (?, ?, ?, ?)",
                 (username, datetime.now(), question, answer))
        conn.commit()
    except Exception as e:
        st.error(f"Database error: {str(e)}")
    finally:
        conn.close()

@timeit
def get_user_history(username, limit=50):
    """Retrieve user's chat history"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute("""
            SELECT question, answer, timestamp 
            FROM chat_history 
            WHERE username = ? 
            ORDER BY timestamp DESC
            LIMIT ?
        """, (username, limit))
        return c.fetchall()
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return []
    finally:
        conn.close()

@timeit
def clear_user_history(username):
    """Clear user's chat history"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute("DELETE FROM chat_history WHERE username = ?", (username,))
        conn.commit()
        return c.rowcount
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return 0
    finally:
        conn.close()