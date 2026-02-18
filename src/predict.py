# src/predict.py
import joblib
import numpy as np
import pandas as pd
from src.config import MODEL_PATH

def load_model_bundle():
    return joblib.load(MODEL_PATH)

def predict_price(bundle, input_dict):
    pipe = bundle["pipeline"]
    features = bundle["features_used"]
    
    # تأمين المدخلات
    row = {k: input_dict.get(k, 0 if isinstance(v, (int, float)) else "Unknown") 
           for k, v in input_dict.items() if k in features}
    
    X = pd.DataFrame([row])
    pred = pipe.predict(X)[0]
    return float(np.expm1(pred)) # إعادة القيمة من لوغاريتم لدولار