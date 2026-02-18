import json
import sqlite3
import pandas as pd
from datetime import datetime
from src.config import LOG_DB_PATH, LOG_DIR

def _connect():
    """تأكيد وجود المجلد وفتح الاتصال بقاعدة البيانات."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(LOG_DB_PATH)

def init_db():
    """إنشاء الجدول بتصميم يدعم التحليل المتقدم."""
    with _connect() as con:
        cur = con.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            model_type TEXT,
            use_log_target INTEGER,
            features_json TEXT,
            predicted_price REAL,
            listed_price REAL,
            diff_amount REAL, -- ميزة إضافية: الفرق بين السعرين
            deal_label TEXT
        )
        """)
        con.commit()

def log_prediction(model_type: str, use_log_target: bool, features: dict,
                   predicted_price: float, listed_price: float, deal_label: str):
    """تسجيل العملية مع حساب الفرق السعري تلقائياً."""
    init_db()
    
    # حساب الفرق لسهولة التحليل لاحقاً
    diff_amount = float(predicted_price - listed_price)
    
    with _connect() as con:
        cur = con.cursor()
        cur.execute("""
        INSERT INTO predictions(created_at, model_type, use_log_target, features_json,
                                predicted_price, listed_price, diff_amount, deal_label)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # تنسيق وقت أسهل للقراءة
            model_type,
            1 if use_log_target else 0,
            json.dumps(features, ensure_ascii=False),
            float(predicted_price),
            float(listed_price),
            diff_amount,
            str(deal_label)
        ))
        con.commit()

def read_logs() -> pd.DataFrame:
    """قراءة السجلات وإرجاعها كـ DataFrame لتسهيل عرضها في analytics."""
    init_db()
    with _connect() as con:
        # استخدام Pandas للقراءة مباشرة يجعل التعامل مع البيانات أسهل بكثير
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", con)
    return df