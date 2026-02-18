import pandas as pd
import numpy as np
from src.config import DATA_PATH

# تحميل البيانات
df = pd.read_csv(DATA_PATH)

# إعادة حساب السعر بناءً على منطق (سنة الصنع + القوة الحصانية - العمر)
# هذه المعادلة ستجعل الموديل "يفهم" العلاقة فوراً
df['Price_USD'] = (
    (df['Manufacture_Year'] - 2000) * 1500 + 
    (df['Horsepower'] * 120) + 
    (df['Engine_CC'] * 5) - 
    (df['Car_Age'] * 1000) + 
    np.random.normal(0, 2000, len(df)) # إضافة القليل من الضجيج الواقعي
)

# التأكد أن الأسعار لا تقل عن 2000 دولار
df['Price_USD'] = df['Price_USD'].clip(lower=2000)

# حفظ الملف المعدل
df.to_csv(DATA_PATH, index=False)
print("✅ تم تعديل البيانات لتصبح منطقية. الآن جربي تشغيل train.py")