import pandas as pd

def format_car_card(car_row: pd.Series) -> str:
    """تحويل صف بيانات السيارة إلى بطاقة نصية جميلة للتشات بوت."""
    emoji_map = {
        "SUV": "🚙",
        "Sedan": "🚗",
        "Coupe": "🏎️",
        "Hatchback": "🚗",
        "Pickup": "🛻"
    }
    
    body_emoji = emoji_map.get(car_row.get("Body_Type", ""), "🚘")
    
    card = (
        f"{body_emoji} **{car_row['Brand']} {int(car_row['Year'])}**\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"💰 **السعر:** ${car_row['Price_USD']:,}\n"
        f"⚙️ **المحرك:** {int(car_row['Engine_CC'])} CC | {int(car_row['Horsepower'])} HP\n"
        f"⛽ **الوقود:** {car_row['Fuel_Type']} ({car_row['Transmission']})\n"
        f"🛣️ **المسافة:** {car_row['Mileage_km_per_l']} كم/لتر\n"
        f"⏳ **العمر:** {int(car_row['Car_Age'])} سنوات\n"
    )
    return card

def display_deal_badge(label: str, confidence: float) -> str:
    """إنشاء شعار (Badge) لتقييم الصفقة مع نسبة الثقة."""
    color_emoji = "🟢" if "ممتازة" in label else "🟡" if "عادل" in label else "🔴"
    
    badge = (
        f"\n{color_emoji} **التقييم:** {label}\n"
        f"🎯 **ثقة النموذج:** {confidence}%\n"
    )
    return badge

def create_stats_table(kpis: dict) -> str:
    """عرض إحصائيات سريعة للبيانات."""
    table = (
        f"📊 **إحصائيات السوق الحالية:**\n"
        f"• عدد السيارات المتاحة: {kpis['count']}\n"
        f"• متوسط الأسعار: ${kpis['mean_price']:,.0f}\n"
        f"• السعر المتوقع (الوسيط): ${kpis['median_price']:,.0f}\n"
    )
    return table