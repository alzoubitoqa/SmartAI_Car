from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from src.predict import load_model_bundle, predict_price
from src.deal import evaluate_deal
from src.config import MODEL_PATH

# 1. إنشاء التطبيق
app = FastAPI(
    title="SmartCar AI API",
    description="واجهة برمجة تطبيقات لتوقع أسعار السيارات وتقييم الصفقات بناءً على الذكاء الاصطناعي",
    version="1.0.0"
)

# 2. تحميل الموديل عند التشغيل لضمان السرعة
try:
    bundle = load_model_bundle()
except Exception as e:
    bundle = None
    print(f"⚠️ تحذير: فشل تحميل الموديل. تأكد من تشغيل train.py أولاً. الخطأ: {e}")

# 3. تعريف نموذج البيانات المدخلة (Schema)
class CarRequest(BaseModel):
    brand: str
    body_type: str
    year: int
    horsepower: float
    engine_cc: float
    fuel_type: str
    transmission: str
    listed_price: float = 0.0  # اختياري لتقييم الصفقة

# 4. نقطة النهاية (Endpoints)
@app.get("/")
def read_root():
    return {"status": "online", "message": "SmartCar AI API is running successfully"}

@app.post("/predict")
def get_prediction(car: CarRequest):
    if bundle is None:
        raise HTTPException(status_code=500, detail="Model bundle not loaded on server")

    # تجهيز البيانات المدخلة لتناسب الموديل
    input_data = {
        "Brand": car.brand,
        "Body_Type": car.body_type,
        "Year": car.year,
        "Horsepower": car.horsepower,
        "Engine_CC": car.engine_cc,
        "Fuel_Type": car.fuel_type,
        "Transmission": car.transmission,
        "Car_Age": 2026 - car.year,
        "HP_per_CC": car.horsepower / (car.engine_cc + 1),
        "Mileage_km_per_l": 15.0
    }

    try:
        # التوقع
        predicted_price = predict_price(bundle, input_data)
        
        # التقييم (في حال تم تزويدنا بسعر معروض)
        deal_info = None
        if car.listed_price > 0:
            deal = evaluate_deal(car.listed_price, predicted_price, bundle['metrics']['mae'], bundle['metrics']['r2'])
            deal_info = {
                "label": deal.label,
                "fair_range": {"lower": round(deal.lower, 2), "upper": round(deal.upper, 2)},
                "confidence_score": f"{deal.confidence_score}%"
            }

        return {
            "car_details": car,
            "ai_predicted_price": round(predicted_price, 2),
            "deal_analysis": deal_info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"خطأ أثناء المعالجة: {str(e)}")

# لتشغيل السيرفر محلياً
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)