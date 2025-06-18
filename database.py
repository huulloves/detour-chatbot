import sqlite3

def get_connection(db_path="chatbot.db"):
    return sqlite3.connect(db_path)

def setup_database(conn):
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            user_input TEXT,
            bot_response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()

def log_conversation(conn, user_id, user_input, bot_response):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO conversations (user_id, user_input, bot_response)
        VALUES (?, ?, ?)
    ''', (user_id, user_input, bot_response))
    conn.commit()

def get_conversations(conn, user_id):
    cursor = conn.cursor()
    cursor.execute('''
        SELECT user_input, bot_response, timestamp FROM conversations
        WHERE user_id = ?
        ORDER BY timestamp DESC
    ''', (user_id,))
    return cursor.fetchall()
