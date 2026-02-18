import pandas as pd
import numpy as np
from pathlib import Path

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # توحيد السنة وحساب العمر
    if "Year" not in df.columns and "Manufacture_Year" in df.columns:
        df["Year"] = df["Manufacture_Year"]
    
    df["Car_Age"] = 2026 - df["Year"]

    # هندسة الميزات (تساعد جداً في رفع R2)
    df["HP_per_CC"] = df["Horsepower"] / (df["Engine_CC"] + 1)

    # تنظيف الأسعار المتطرفة (Outliers)
    q_low = df["Price_USD"].quantile(0.01)
    q_hi  = df["Price_USD"].quantile(0.99)
    df = df[(df["Price_USD"] < q_hi) & (df["Price_USD"] > q_low)]

    return df