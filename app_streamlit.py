import streamlit as st
import pandas as pd
import joblib
import datetime
from data_processor import WeatherDataProcessor

# 1. Page Configuration
st.set_page_config(page_title="SkyCast AI", page_icon="🌈", layout="wide")

# --- CUSTOM CSS (Colorful & Modern) ---
st.markdown("""
    <style>
    /* خلفية متدرجة مبهجة */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* تنسيق الحاوية الرئيسية */
    .stForm {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        border: 1px solid #e1e8ed;
    }

    /* عناوين ملونة */
    h1 {
        color: #2e86de !important;
        text-align: center;
        font-family: 'Trebuchet MS', sans-serif;
    }
    
    /* تنسيق الـ Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f2f6;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        color: #2f3542;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2e86de !important;
        color: white !important;
    }

    /* زر التوقع الملون */
    .stButton>button {
        background: linear-gradient(to right, #2e86de, #00d2ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 15px !important;
        font-size: 18px !important;
        font-weight: bold !important;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(46, 134, 222, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Load Tools
@st.cache_resource
def load_prediction_tools():
    model = joblib.load('models/rain_forest_model.pkl')
    proc = WeatherDataProcessor()
    proc.load_assets('models/') 
    return model, proc

model, processor = load_prediction_tools()

# 3. Header
st.title("🌈 SkyCast AI: Premium Weather Predictor")
st.markdown("<p style='text-align: center; color: #57606f;'>Smart insights for a brighter tomorrow.</p>", unsafe_allow_html=True)

# 4. Input Form
with st.form("colorful_form"):
    # الجزء العلوي: المدينة والتاريخ
    col_top1, col_top2 = st.columns(2)
    with col_top1:
        location = st.selectbox("📍 Target Location", processor.le.classes_)
    with col_top2:
        date = st.date_input("📅 Forecast Date", value=datetime.date(2020, 12, 31),
                            min_value=datetime.date(2008, 1, 1), 
                            max_value=datetime.date(2020, 12, 31))

    # تنظيم المدخلات في Tabs ملونة
    tab1, tab2, tab3 = st.tabs(["🌡️ Temperature & Rain", "💧 Humidity & Pressure", "💨 Wind & Sky"])

    with tab1:
        st.info("Set the temperature and current rainfall levels.")
        c1, c2 = st.columns(2)
        min_temp = c1.number_input("Min Temp (°C)", value=15.0)
        max_temp = c2.number_input("Max Temp (°C)", value=25.0)
        
        c3, c4 = st.columns(2)
        rainfall = c3.number_input("Rainfall Today (mm)", value=0.0)
        rain_today = c4.selectbox("Did it rain today?", ["No", "Yes"])

    with tab2:
        st.info("Fine-tune the moisture and atmospheric pressure.")
        c5, c6 = st.columns(2)
        hum_3pm = c5.slider("3pm Humidity (%)", 0, 100, 50)
        hum_9am = c6.slider("9am Humidity (%)", 0, 100, 55)
        
        c7, c8 = st.columns(2)
        pres_3pm = c7.number_input("3pm Pressure (hPa)", value=1010.0)
        pres_9am = c8.number_input("9am Pressure (hPa)", value=1015.0)

    with tab3:
        st.info("Check the wind patterns and sky coverage.")
        c9, c10 = st.columns(2)
        wind_gust = c9.number_input("Wind Gust Speed (km/h)", value=40.0)
        wind_dir = c10.selectbox("Wind Direction", list(processor.compass_map.keys()))
        
        c11, c12 = st.columns(2)
        sunshine = c11.slider("Sunshine (Hours)", 0.0, 15.0, 7.0)
        cloud = c12.slider("Cloud Cover (3pm)", 0, 9, 4)

    # القيم الافتراضية الذكية
    temp_9am = (min_temp + max_temp) / 2
    temp_3pm = max_temp - 1
    evap = 4.0
    wind_9am_speed = 15.0
    wind_3pm_speed = 20.0
    cloud_9am = 4

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.form_submit_button("ANALYZE WEATHER")

# 5. Handling Prediction
if submit:
    user_input = {
        'Date': date, 'Location': location, 'MinTemp': min_temp, 'MaxTemp': max_temp,
        'Rainfall': rainfall, 'Evaporation': evap, 'Sunshine': sunshine, 
        'WindGustDir': wind_dir, 'WindGustSpeed': wind_gust, 
        'WindDir9am': wind_dir, 'WindDir3pm': wind_dir, 
        'WindSpeed9am': wind_9am_speed, 'WindSpeed3pm': wind_3pm_speed, 
        'Humidity9am': hum_9am, 'Humidity3pm': hum_3pm, 
        'Pressure9am': pres_9am, 'Pressure3pm': pres_3pm,
        'Cloud9am': cloud_9am, 'Cloud3pm': cloud, 
        'Temp9am': temp_9am, 'Temp3pm': temp_3pm, 'RainToday': rain_today
    }
    
    df_input = pd.DataFrame([user_input])
    processed_df = processor.clean_data(df_input, is_training=False)
    X_final = processed_df.drop(['Date', 'month'], axis=1, errors='ignore')
    
    prediction = model.predict(X_final)[0]
    prob = model.predict_proba(X_final)[0][1]

    # عرض النتائج بطريقة بصرية ملونة
    st.markdown("---")
    res_col1, res_col2 = st.columns([2, 1])

    with res_col1:
        if prediction == 1:
            st.markdown(f"""
                <div style="background-color: #ff4757; padding: 20px; border-radius: 15px; color: white;">
                    <h2>⛈️ It will Rain Tomorrow!</h2>
                    <p>High probability of precipitation detected by our AI engine.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background-color: #2ed573; padding: 20px; border-radius: 15px; color: white;">
                    <h2>☀️ Clear Skies Ahead!</h2>
                    <p>Our model predicts stable conditions for a dry tomorrow.</p>
                </div>
                """, unsafe_allow_html=True)

    with res_col2:
        confidence = prob if prediction == 1 else (1 - prob)
        st.metric("Confidence Level", f"{confidence:.1%}")
        st.progress(confidence)