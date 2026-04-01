import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
 
import pandas as pd
 
# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH        = "fraud_predictions.db"
REFERENCE_PATH = "reference_data.parquet"
PIPELINE_PATH  = "fraud_pipeline.pkl"
REPORTS_DIR    = Path("monitoring_reports")
REPORTS_DIR.mkdir(exist_ok=True)
 
# Alert thresholds — tune these to your business tolerance
ALERT_THRESHOLDS = {
    "recall_min"         : 0.85,   # alert if recall drops below this
    "precision_min"      : 0.30,   # alert if precision drops below this
    "fraud_rate_max"     : 0.05,   # alert if live fraud rate exceeds 5%
    "drift_share_max"    : 0.30,   # alert if >30% of features are drifting
    "missing_values_max" : 0.01,   # alert if >1% of values are missing
}

def init_db(db_path: str = DB_PATH):
    """Create the predictions table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT    NOT NULL,
            step             INTEGER,
            type             TEXT,
            amount           REAL,
            oldbalanceOrg    REAL,
            oldbalanceDest   REAL,
            hour             INTEGER,
            day              INTEGER,
            fraud_probability REAL,
            predicted_fraud  INTEGER,
            actual_fraud     INTEGER,   -- NULL until label arrives
            threshold        REAL
        )
    """)
    conn.commit()
    conn.close()

def log_prediction(
    raw_transaction: dict,
    fraud_probability: float,
    predicted_fraud: bool,
    threshold: float,
    actual_fraud: int = None,   # provide later when ground truth arrives
    db_path: str = DB_PATH,
):
    """
    Log one prediction. Call this every time your pipeline runs.
 
    actual_fraud can be None at prediction time — update it later
    via update_actual_label() once the transaction is confirmed.
    """
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        INSERT INTO predictions
            (timestamp, step, type, amount, oldbalanceOrg, oldbalanceDest,
             hour, day, fraud_probability, predicted_fraud, actual_fraud, threshold)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.utcnow().isoformat(),
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
    conn.close()

def update_actual_label(prediction_id: int, actual_fraud: int,
                         db_path: str = DB_PATH):
    """
    Update the ground-truth label for a logged prediction.
    In a real system you'd call this when a fraud case is confirmed
    by an analyst or chargeback system.
    """
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE predictions SET actual_fraud = ? WHERE id = ?",
        (actual_fraud, prediction_id)
    )
    conn.commit()
    conn.close()
 
 
def load_predictions(days: int = 7, db_path: str = DB_PATH) -> pd.DataFrame:
    """Load predictions from the last N days."""
    init_db(db_path)
    conn  = sqlite3.connect(db_path)
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()
    df    = pd.read_sql(
        "SELECT * FROM predictions WHERE timestamp >= ? ORDER BY timestamp",
        conn, params=(since,)
    )
    conn.close()
    return df