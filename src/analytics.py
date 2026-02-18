import pandas as pd

def dataset_kpis(df: pd.DataFrame) -> dict:
    """
    حساب المؤشرات الرئيسية للأداء مع إضافة مقاييس التشتت
    للمساعدة في فهم جودة البيانات.
    """
    return {
        "count": int(len(df)),
        "mean_price": float(df["Price_USD"].mean()),
        "median_price": float(df["Price_USD"].median()), 
        "min_price": float(df["Price_USD"].min()),
        "max_price": float(df["Price_USD"].max()),
        "std_dev_price": float(df["Price_USD"].std()), 
    }

def price_by_brand(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    تحليل متوسط السعر حسب الماركة مع حساب عدد السيارات لكل ماركة
    للتأكد من أن الإحصائية مبنية على عدد كافٍ من البيانات.
    """
    return (
        df.groupby("Brand", dropna=True)["Price_USD"]
        .agg(["mean", "count"]) 
        .sort_values(by="mean", ascending=False)
        .head(top_n)
        .reset_index()
        .rename(columns={"mean": "Avg_Price_USD", "count": "Car_Count"})
    )

def price_by_body(df: pd.DataFrame) -> pd.DataFrame:
    """تحليل متوسط السعر حسب شكل جسم السيارة."""
    return (
        df.groupby("Body_Type", dropna=True)["Price_USD"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"Price_USD": "Avg_Price_USD"})
    )

def correlation_analysis(df: pd.DataFrame) -> pd.Series:
    """
    ميزة جديدة: تحليل الارتباط (Correlation)
    هذا يخبرك أي ميزة (مثل HP أو Age) تؤثر أكثر في السعر.
    """
    numeric_df = df.select_dtypes(include=['number'])
    return numeric_df.corr()["Price_USD"].sort_values(ascending=False)

def price_by_age_bracket(df: pd.DataFrame) -> pd.DataFrame:
    """
    ميزة جديدة: تحليل السعر بناءً على فئات العمر.
    """
    if "Car_Age" not in df.columns:
        return pd.DataFrame()
        
    bins = [0, 3, 8, 15, 100]
    labels = ["New (0-3y)", "Modern (4-8y)", "Used (9-15y)", "Classic (>15y)"]
    df['Age_Bracket'] = pd.cut(df['Car_Age'], bins=bins, labels=labels)
    
    return (
        df.groupby("Age_Bracket")["Price_USD"]
        .mean()
        .reset_index()
        .rename(columns={"Price_USD": "Avg_Price_USD"})
    )