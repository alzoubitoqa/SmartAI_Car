from dataclasses import dataclass

@dataclass
class DealResult:
    label: str
    lower: float
    upper: float
    confidence_score: float

def evaluate_deal(listed_price, predicted_price, mae_usd, r2_score=0.0):
    # نطاق مرن يعتمد على دقة الموديل
    band = max(0.07 * predicted_price, 0.8 * mae_usd)
    lower, upper = predicted_price - band, predicted_price + band
    
    confidence = round(max(0, r2_score * 100), 2)

    if listed_price < lower:
        label = "🔥 صفقة ممتازة (Great Deal)"
    elif listed_price > upper:
        label = "⚠️ مبالغ فيه (Overpriced)"
    else:
        label = "✅ سعر عادل (Fair Price)"

    return DealResult(label, lower, upper, confidence)