import sqlite3
import os
import yaml

def load_config(config_path: str):
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError as e:
        print(f"Config file not found: {e}")
        return None

def init_prove_database():
    # Load your config directly without importing functions
    config = load_config('config.yaml')
    if not config:
        print("❌ Failed to load config")
        return
    
    db_path = config['database']['result_db_for_API']
    
    print(f"Initializing SQLite database at: {db_path}")
    
    # Create directory if needed
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect to SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create status table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS status (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT UNIQUE,
            qid TEXT,
            status TEXT,
            start_time TEXT,
            algo_version TEXT,
            request_type TEXT
        )
    ''')
    
    # Create aggregated_results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS aggregated_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT,
            qid TEXT,
            property_id TEXT,
            url TEXT,
            triple TEXT,
            result TEXT,
            result_sentence TEXT,
            FOREIGN KEY (task_id) REFERENCES status (task_id)
        )
    ''')
    
    # Create indexes for better performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_status_qid ON status(qid)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_status_task_id ON status(task_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_task_id ON aggregated_results(task_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_qid ON aggregated_results(qid)')
    
    conn.commit()
    
    # Verify tables were created
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("✅ Tables created:", [table[0] for table in tables])
    
    conn.close()
    print(f"✅ ProVe SQLite database initialized at: {db_path}")

if __name__ == "__main__":
    init_prove_database()
