from pathlib import Path

# المسار الرئيسي للمشروع
BASE_DIR = Path(__file__).resolve().parents[1]

# مسارات البيانات والنماذج
DATA_PATH = BASE_DIR / "data" / "cars.csv"
MODEL_PATH = BASE_DIR / "models" / "price_model.pkl"

# مسارات السجلات
LOG_DIR = BASE_DIR / "logs"
LOG_DB_PATH = LOG_DIR / "predictions.db"

# إنشاء المجلدات تلقائياً
LOG_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "models").mkdir(parents=True, exist_ok=True)