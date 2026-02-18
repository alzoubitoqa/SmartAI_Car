import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.config import DATA_PATH, MODEL_PATH
from src.data_loader import load_data
from src.train import train_price_model
from src.predict import load_model_bundle, predict_price
from src.deal import evaluate_deal
from src.logging_db import log_prediction, read_logs
from src.chatbot_rules import parse_user_message, recommend
from src.analytics import dataset_kpis, price_by_brand

# إعدادات الصفحة
st.set_page_config(page_title="SmartCar AI Pro", layout="wide", page_icon="🏎️")

# تحميل البيانات مع التخزين المؤقت للسرعة
@st.cache_data
def get_cached_data():
    return load_data(DATA_PATH)

# التحقق من وجود البيانات
try:
    df = get_cached_data()
except Exception as e:
    st.error(f"❌ لم يتم العثور على ملف البيانات: {e}")
    st.stop()

st.title("🏎️ SmartCar AI Professional Marketplace")

# القائمة الجانبية (Sidebar)
st.sidebar.header("⚙️ التحكم بالنظام")
if st.sidebar.button("🚀 إعادة تدريب الموديل"):
    with st.sidebar.status("جاري التدريب..."):
        # تم تعديل المفتاح هنا ليناسب ملف الـ train الخاص بك
        res = train_price_model(df)
        st.sidebar.success(f"تم بنجاح! R²: {res['metrics']['r2']:.4f}")

# التبويبات الرئيسية
tabs = st.tabs(["📊 Dashboard", "🔍 Car Discovery", "💰 AI Valuator", "🤖 Chatbot Assistant", "📜 Logs"])

# --- Tab 1: Dashboard ---
with tabs[0]:
    st.subheader("📊 تحليل بيانات السوق")
    kpis = dataset_kpis(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("إجمالي السيارات", f"{kpis['count']:,}")
    c2.metric("متوسط السعر", f"${kpis['mean_price']:,.0f}")
    c3.metric("أقل سعر", f"${kpis['min_price']:,.0f}")
    c4.metric("أعلى سعر", f"${kpis['max_price']:,.0f}")

    col_graph1, col_graph2 = st.columns(2)
    with col_graph1:
        brand_data = price_by_brand(df)
        fig = px.bar(brand_data, x="Brand", y="Avg_Price_USD", title="متوسط السعر حسب الماركة", color="Avg_Price_USD")
        st.plotly_chart(fig, use_container_width=True)
    with col_graph2:
        # رسم بياني يوضح العلاقة التي حققت R2 عالية
        fig2 = px.scatter(df, x="Horsepower", y="Price_USD", color="Body_Type", hover_data=['Year'], title="العلاقة بين القوة والسعر")
        st.plotly_chart(fig2, use_container_width=True)

# --- Tab 2: Discovery ---
with tabs[1]:
    st.subheader("🔎 استكشاف وتصفية السيارات")
    f_brand = st.multiselect("اختر الماركة", sorted(df["Brand"].unique()))
    f_price = st.slider("نطاق السعر ($)", int(df["Price_USD"].min()), int(df["Price_USD"].max()), (10000, 50000))
    
    filtered_df = df.copy()
    if f_brand: filtered_df = filtered_df[filtered_df["Brand"].isin(f_brand)]
    filtered_df = filtered_df[(filtered_df["Price_USD"] >= f_price[0]) & (filtered_df["Price_USD"] <= f_price[1])]
    st.dataframe(filtered_df, use_container_width=True)

# --- Tab 3: Valuator ---
with tabs[2]:
    st.subheader("💰 المقيم الذكي (AI Valuator)")
    if not MODEL_PATH.exists():
        st.warning("⚠️ الموديل غير موجود! يرجى الضغط على 'إعادة تدريب' من القائمة الجانبية.")
    else:
        bundle = load_model_bundle()
        col_in1, col_in2 = st.columns(2)
        with col_in1:
            in_brand = st.selectbox("الماركة", sorted(df["Brand"].unique()))
            in_body = st.selectbox("نوع الجسم", sorted(df["Body_Type"].unique())) # حل مشكلة KeyError: Body_Type
            in_year = st.number_input("سنة الصنع", 1990, 2026, 2022)
        with col_in2:
            in_hp = st.number_input("القوة الحصانية (HP)", 50, 1000, 200)
            in_cc = st.number_input("سعة المحرك (CC)", 800, 7000, 2000)
            in_fuel = st.selectbox("الوقود", df["Fuel_Type"].unique())
            in_trans = st.selectbox("ناقل الحركة", df["Transmission"].unique())
        
        in_listed = st.number_input("السعر المعروض حالياً ($)", value=25000)

        if st.button("⚖️ تحليل القيمة العادلة"):
            # تجهيز الميزات بنفس الترتيب والمسميات التي تدرب عليها الموديل
            input_feats = {
                "Brand": in_brand, 
                "Body_Type": in_body,
                "Year": in_year, 
                "Horsepower": in_hp,
                "Engine_CC": in_cc, 
                "Fuel_Type": in_fuel, 
                "Transmission": in_trans,
                "Car_Age": 2026 - in_year, 
                "HP_per_CC": in_hp / (in_cc + 1),
                "Mileage_km_per_l": 15.0
            }
            
            pred = predict_price(bundle, input_feats)
            # استخدام مفاتيح bundle الصحيحة للتقييم
            deal = evaluate_deal(in_listed, pred, bundle['metrics']['mae'], bundle['metrics']['r2'])
            
            st.divider()
            res_c1, res_c2 = st.columns(2)
            with res_c1:
                st.metric("سعر الذكاء الاصطناعي المتوقع", f"${pred:,.0f}")
                st.write(f"🎯 ثقة النموذج: **{deal.confidence_score}%**")
            with res_c2:
                st.subheader(f"النتيجة: {deal.label}")
                st.info(f"نطاق السعر العادل: **${deal.lower:,.0f} - ${deal.upper:,.0f}**")
            
            log_prediction("RandomForest", True, input_feats, pred, in_listed, deal.label)

# --- Tab 4: Chatbot ---
# --- Tab 4: Chatbot Assistant ---
with tabs[3]:
    st.subheader("🤖 مساعد الشراء الذكي")
    st.write("اكتب ما تبحث عنه، مثلاً: 'بدي سيارة تويوتا تحت الـ 30000' أو 'Kia 2022 Petrol'")
    
    # استخدام st.form لمنع الـ App من إعادة التحميل عند كل حرف
    with st.form(key='chat_form'):
        chat_input = st.text_input("أدخل طلبك هنا:")
        submit_button = st.form_submit_button(label='بحث ذكي 🔍')

    if submit_button and chat_input:
        with st.spinner("جاري تحليل طلبك والبحث في قاعدة البيانات..."):
            # 1. تحليل الجملة
            prefs = parse_user_message(chat_input)
            
            # 2. جلب التوصيات
            recs = recommend(df, prefs)
            
            if recs is not None and not recs.empty:
                st.success(f"✅ وجدت لك هذه الخيارات الرائعة:")
                
                # عرض النتائج بشكل جميل
                for _, car in recs.iterrows():
                    with st.expander(f"🏎️ {car['Brand']} {int(car['Year'])} - ${car['Price_USD']:,.0f}"):
                        c1, c2 = st.columns(2)
                        c1.write(f"**نوع الجسم:** {car['Body_Type']}")
                        c1.write(f"**ناقل الحركة:** {car['Transmission']}")
                        c2.write(f"**نوع الوقود:** {car['Fuel_Type']}")
                        c2.write(f"**القوة الحصانية:** {car['Horsepower']} HP")
            else:
                st.warning("⚠️ لم أجد تطابقاً دقيقاً. جربي تغيير البحث (مثلاً: اذكر السعر أو الماركة فقط).")

# --- Tab 5: Logs ---
with tabs[4]:
    st.subheader("📜 سجل العمليات (Logs)")
    try:
        logs_df = read_logs()
        st.dataframe(logs_df, use_container_width=True)
    except:
        st.write("لا يوجد سجلات متاحة حالياً.")