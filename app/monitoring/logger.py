import os
import psycopg2
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")   # from Render
REFERENCE_PATH = "reference_data.parquet"
PIPELINE_PATH  = "fraud_pipeline.pkl"
REPORTS_DIR    = Path("monitoring_reports")
REPORTS_DIR.mkdir(exist_ok=True)

# ── Alert thresholds ──────────────────────────────────────────────────────────
ALERT_THRESHOLDS = {
    "recall_min"         : 0.85,
    "precision_min"      : 0.30,
    "fraud_rate_max"     : 0.05,
    "drift_share_max"    : 0.30,
    "missing_values_max" : 0.01,
}

# ── DB CONNECTION ─────────────────────────────────────────────────────────────
def get_connection():
    return psycopg2.connect(DATABASE_URL)

# ── INIT DB ───────────────────────────────────────────────────────────────────
def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            timestamp TEXT NOT NULL,
            step INTEGER,
            type TEXT,
            amount FLOAT,
            oldbalanceOrg FLOAT,
            oldbalanceDest FLOAT,
            hour INTEGER,
            day INTEGER,
            fraud_probability FLOAT,
            predicted_fraud INTEGER,
            actual_fraud INTEGER,
            threshold FLOAT
        )
    """)

    conn.commit()
    cur.close()
    conn.close()

# ── LOG PREDICTION ────────────────────────────────────────────────────────────
def log_prediction(
    raw_transaction: dict,
    fraud_probability: float,
    predicted_fraud: bool,
    threshold: float,
    actual_fraud: int = None,
):
    init_db()

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO predictions
        (timestamp, step, type, amount, oldbalanceOrg, oldbalanceDest,
         hour, day, fraud_probability, predicted_fraud, actual_fraud, threshold)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        datetime.now(timezone.utc).isoformat(),
        raw_transaction.get("step"),
        raw_transaction.get("type"),
        raw_transaction.get("amount"),
        raw_transaction.get("oldbalanceOrg"),
        raw_transaction.get("oldbalanceDest"),
        int(raw_transaction.get("step", 0)) % 24,
        int(raw_transaction.get("step", 0)) // 24,
        round(float(fraud_probability), 6),
        int(predicted_fraud),
        actual_fraud,
        float(threshold),
    ))

    conn.commit()
    cur.close()
    conn.close()

# ── UPDATE LABEL ──────────────────────────────────────────────────────────────
def update_actual_label(prediction_id: int, actual_fraud: int):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "UPDATE predictions SET actual_fraud = %s WHERE id = %s",
        (actual_fraud, prediction_id)
    )

    conn.commit()
    cur.close()
    conn.close()

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
def load_predictions(days: int = 7) -> pd.DataFrame:
    init_db()

    conn = get_connection()
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    query = """
        SELECT * FROM predictions
        WHERE timestamp >= %s
        ORDER BY timestamp
    """

    df = pd.read_sql(query, conn, params=(since,))
    conn.close()

    return df