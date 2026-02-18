import pandas as pd
from src.data_loader import load_data
from src.config import DATA_PATH

df = load_data(DATA_PATH)
# حساب الارتباط الرقمي
correlations = df.select_dtypes(include=['number']).corr()['Price_USD'].sort_values(ascending=False)
print("📊 قوة ارتباط المواصفات بالسعر (يجب أن تكون بعيدة عن الصفر):")
print(correlations)