import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error

# استيراد الإعدادات والميزات من الملفات التي أنشأناها
from src.config import DATA_PATH, MODEL_PATH
from src.features import FEATURES_NUMERIC, FEATURES_CATEGORICAL, TARGET_COLUMN
from src.data_loader import load_data

def train_price_model(df=None):
    """
    تدريب النموذج باستخدام خوارزمية Random Forest مع معالجة متقدمة للبيانات.
    """
    print("⏳ جاري تحضير البيانات...")
    if df is None:
        df = load_data(DATA_PATH)

    # 1. فصل الميزات عن الهدف
    X = df[FEATURES_NUMERIC + FEATURES_CATEGORICAL]
    y = df[TARGET_COLUMN]

    # 2. السر في رفع R2: تحويل السعر للوغاريتم لتقليل أثر الفجوات السعرية الكبيرة
    y_log = np.log1p(y)

    # 3. بناء معالج البيانات (Preprocessing)
    # StandardScaler للأرقام و OneHotEncoder للنصوص
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, FEATURES_NUMERIC),
            ('cat', categorical_transformer, FEATURES_CATEGORICAL)
        ])

    # 4. بناء النموذج (استخدام RandomForestRegressor بدلاً من الموديلات الخطية)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=300,   # عدد الأشجار
            max_depth=20,       # العمق الأقصى لمنع Overfitting
            min_samples_split=5,
            random_state=42     # لضمان ثبات النتائج عند كل تشغيل
        ))
    ])

    # 5. تقسيم البيانات (80% تدريب، 20% اختبار)
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

    print(f"🚀 جاري التدريب على {len(X_train)} عينة...")
    model.fit(X_train, y_train)

    # 6. التقييم
    y_pred_log = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_log)
    
    # حساب الخطأ الحقيقي بالدولار (عبر عكس اللوغاريتم)
    actual_prices = np.expm1(y_test)
    predicted_prices = np.expm1(y_pred_log)
    mae = mean_absolute_error(actual_prices, predicted_prices)

    print("\n" + "="*30)
    print(f"✅ انتهى التدريب بنجاح!")
    print(f"📊 دقة النموذج (R² Score): {r2:.4f}")
    print(f"💰 متوسط الخطأ المطلق: {mae:,.2f} دولار")
    print("="*30)

    # 7. حفظ النموذج (Bundle)
    bundle = {
        "pipeline": model,
        "features_used": FEATURES_NUMERIC + FEATURES_CATEGORICAL,
        "metrics": {"r2": r2, "mae": mae},
        "use_log_target": True
    }
    
    joblib.dump(bundle, MODEL_PATH)
    print(f"💾 تم حفظ النموذج في: {MODEL_PATH}")
    
    return bundle

if __name__ == "__main__":
    train_price_model()