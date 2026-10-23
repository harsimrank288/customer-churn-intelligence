
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "churn_history.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def create_table():
    connection = get_connection()

    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_data TEXT NOT NULL,
                predicted_churn TEXT NOT NULL,
                churn_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.commit()

    finally:
        connection.close()


def save_prediction(
    customer_data,
    predicted_churn,
    churn_probability,
    risk_level
):
    connection = get_connection()

    try:
        cursor = connection.execute("""
            INSERT INTO predictions (
                customer_data,
                predicted_churn,
                churn_probability,
                risk_level
            )
            VALUES (?, ?, ?, ?)
        """, (
            customer_data,
            predicted_churn,
            churn_probability,
            risk_level
        ))

        connection.commit()

        return cursor.lastrowid

    finally:
        connection.close()

def get_prediction_history():
    connection = get_connection()

    try:
        records = connection.execute("""
            SELECT * FROM predictions
            ORDER BY id DESC
        """).fetchall()

        return [dict(record) for record in records]

    finally:
        connection.close()
