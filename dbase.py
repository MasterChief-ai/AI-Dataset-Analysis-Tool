import os
import psycopg2
import json
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", 5432)
    )

def log_user_input(report_id, target_col, task_type, best_model, metrics, insights, explanation, ai_insight):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS analysis_logs (
            id SERIAL PRIMARY KEY,
            report_id VARCHAR(50),
            target_col TEXT,
            task_type TEXT,
            best_model TEXT,
            metrics JSONB,
            insights TEXT,
            explanation TEXT,
            ai_insight TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    cur.execute("""
        INSERT INTO analysis_logs (report_id, target_col, task_type, best_model, metrics, insights, explanation, ai_insight)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
    """, (report_id, target_col, task_type, best_model, json.dumps(metrics), insights, explanation, ai_insight))

    conn.commit()
    cur.close()
    conn.close()
