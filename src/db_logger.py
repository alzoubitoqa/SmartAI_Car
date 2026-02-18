import sqlite3
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

def init_db(db_path: Path) -> None:
    """إنشاء قاعدة البيانات والجداول إذا لم تكن موجودة."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            model_type TEXT,
            use_log_target INTEGER,
            features_json TEXT,
            predicted_price REAL,
            listed_price REAL,
            diff_amount REAL, -- ميزة جديدة: فرق السعر بالدولار
            deal_label TEXT
        )
        """)
        conn.commit()

def log_prediction(
    db_path: Path,
    model_type: str,
    use_log_target: bool,
    features: dict, 
    predicted_price: float,
    listed_price: float,
    deal_label: str
) -> None:
    """تسجيل عملية التوقع مع حساب الفروقات السعرية."""
    init_db(db_path)
    
    diff_amount = float(predicted_price - listed_price)
    
    features_json = json.dumps(features, ensure_ascii=False)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO prediction_logs
            (created_at, model_type, use_log_target, features_json, 
             predicted_price, listed_price, diff_amount, deal_label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                model_type,
                1 if use_log_target else 0,
                features_json,
                float(predicted_price),
                float(listed_price),
                diff_amount,
                deal_label
            )
        )
        conn.commit()

def read_logs(db_path: Path) -> pd.DataFrame:
    """قراءة السجلات وتحويلها لـ DataFrame للتحليل."""
    if not db_path.exists():
        return pd.DataFrame()
        
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM prediction_logs ORDER BY id DESC",
            conn
        )
    return df