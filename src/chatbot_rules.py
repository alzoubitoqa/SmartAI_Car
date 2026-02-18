import re
import pandas as pd

def parse_user_message(message):
    """
    تحليل جملة المستخدم واستخراج الماركة، السعر، ونوع الوقود.
    يدعم الكلمات باللغتين العربية والإنجليزية.
    """
    message = message.lower()
    prefs = {}

    # 1. استخراج السعر (البحث عن أي رقم كبير في الجملة)
    price_match = re.search(r'(\d+[\d,]*)\s*(k|\$|ألف|دولار|usd)?', message)
    if price_match:
        price_val = price_match.group(1).replace(',', '')
        # إذا ذكر المستخدم "ألف" أو "k" نضرب في 1000
        multiplier = 1000 if any(x in message for x in ['k', 'ألف', 'thousand']) else 1
        prefs['price_max'] = float(price_val) * multiplier

    # 2. الكلمات المفتاحية للوقود
    if any(x in message for x in ['ديزل', 'diesel']): prefs['fuel'] = 'Diesel'
    if any(x in message for x in ['بنزين', 'petrol', 'gasoline']): prefs['fuel'] = 'Petrol'
    if any(x in message for x in ['كهرباء', 'electric', 'ev']): prefs['fuel'] = 'Electric'

    # 3. استخراج سنة الصنع (البحث عن رقم مكون من 4 خانات يبدأ بـ 20 أو 19)
    year_match = re.search(r'\b(20\d{2}|19\d{2})\b', message)
    if year_match:
        prefs['year'] = int(year_match.group(1))

    # 4. الماركة (سنترك دالة recommend تبحث عنها كـ string)
    # نقوم بتنظيف الجملة من الكلمات الشائعة ليبقى اسم الماركة محتملاً
    common_words = ['بدي', 'سيارة', 'تحت', 'سعر', 'أريد', 'car', 'want', 'under', 'price']
    clean_msg = message
    for word in common_words:
        clean_msg = clean_msg.replace(word, '')
    
    prefs['raw_query'] = clean_msg.strip()
    
    return prefs

def recommend(df, prefs, top_k=5):
    """البحث الفعلي في الـ DataFrame بناءً على التفضيلات."""
    if df is None or df.empty:
        return pd.DataFrame()

    results = df.copy()

    # فلترة السعر (الأهم)
    if 'price_max' in prefs:
        results = results[results['Price_USD'] <= prefs['price_max']]

    # فلترة سنة الصنع
    if 'year' in prefs:
        results = results[results['Year'] >= prefs['year']]

    # فلترة الوقود
    if 'fuel' in prefs:
        results = results[results['Fuel_Type'] == prefs['fuel']]

    # البحث عن الماركة في النص المتبقي
    if prefs.get('raw_query'):
        # نبحث إذا كان أي اسم ماركة موجود في الجملة
        brands = df['Brand'].unique()
        found_brand = None
        for b in brands:
            if b.lower() in prefs['raw_query']:
                found_brand = b
                break
        
        if found_brand:
            results = results[results['Brand'] == found_brand]

    # إذا كانت النتائج فارغة جداً، نعيد أفضل السيارات تقييماً ضمن الميزانية
    if results.empty and 'price_max' in prefs:
        return df[df['Price_USD'] <= prefs['price_max']].sort_values(by='Price_USD', ascending=True).head(top_k)

    return results.head(top_k)