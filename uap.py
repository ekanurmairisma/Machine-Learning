import streamlit as st
import joblib
import gdown
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px


df = pd.read_csv('mountains_vs_beaches_preferences.csv')

#Download model dari Google Drive
#gdown.download("https://drive.google.com/uc?id=1-1YdqcTG12tCNCAxdPzZElhONDPxhdrU", "rf_model.pkl", quiet=False)
#gdown.download("https://drive.google.com/uc?id=1CsKKBZgfUv2qpjUm5uoWzcpALkRBa-VK", "xgb_model.pkl", quiet=False)
#gdown.download("https://drive.google.com/uc?id=1-FWZSvWNG_-EGpcmdYQVxFnCY37aDwki", "fnn_model.h5", quiet=False)

# Memuat model yang sudah disimpan
rf_model = joblib.load('model/rf_model.pkl')
xgb_model = joblib.load('model/xgb_model.pkl')
fnn_model = load_model('model/fnn_model.h5')


# Fungsi untuk membuat prediksi berdasarkan model yang dipilih
def make_prediction(model_choice, input_data):
    if model_choice == "Random Forest":
        prediction_proba = rf_model.predict_proba([input_data]) 
        return prediction_proba[0]  
    elif model_choice == "XGBoost":
        prediction_proba = xgb_model.predict_proba([input_data]) 
        return prediction_proba[0] 
    elif model_choice == "Feedforward Neural Network":
        prediction_proba = fnn_model.predict(np.array([input_data]))  
        prob_mountain = prediction_proba[0][0] 
        prob_beach = 1 - prob_mountain  
        return [prob_beach, prob_mountain]   


# Header dan Deskripsi Aplikasi
st.title('Prediksi Preferensi Liburan: Gunung atau Pantai')
st.write("Aplikasi berbasis web ini digunakan untuk memprediksi preferensi tempat liburan antara Gunung dan Pantai berdasarkan data input dari pengguna. Pengguna juga dapat memilih model prediksi yang ingin digunakan, seperti Random Forest, XGBoost atau Feedforward Neural Network (FNN) untuk mendapatkan hasil prediksi.")
col1, col2 = st.columns([3, 3])  
with col1:
    st.image('assets/gunung.jpg', caption='Gunung')  
with col2:
    st.image('assets/pantai.jpg', caption='Pantai')  

total_responden = len(df)
st.markdown(f"**Total Keseluruhan Responden**: {total_responden}")

st.subheader("Analisis Tempat Liburan")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Distribusi Preferensi**")
    preference_counts = df['Preference'].value_counts()
    plt.figure(figsize=(4, 4))  # Ukuran figure yang sesuai
    plt.pie(preference_counts, 
            labels=['Beach (0)', 'Mountain (1)'], 
            autopct='%1.1f%%', 
            startangle=90, 
            colors=['#2196F3', '#4CAF50'], 
            textprops={'fontsize': 8})  # Sesuaikan warna dan font
    plt.axis('equal')
    st.pyplot(plt)

with col2:
    st.markdown("**Preferensi Berdasarkan Gender**")
    gender_preference = df.groupby('Gender')['Preference'].value_counts().unstack()
    st.bar_chart(gender_preference)

col3, col4 = st.columns(2)
with col3:
    st.markdown("**Travel Frequency**")
    travel_frequency = df['Travel_Frequency'].value_counts()
    st.bar_chart(travel_frequency)

with col4:
    activities = ' '.join(df['Preferred_Activities'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(activities)
    st.markdown("**Word Cloud Aktivitas yang Disukai**")
    st.image(wordcloud.to_array())

col5, col6 = st.columns(2)
with col5:
    st.markdown("**Faktor Musim**")
    season_counts = df['Favorite_Season'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(season_counts, labels=season_counts.index, autopct='%1.1f%%', startangle=90)
    st.pyplot(plt)

with col6:
    location = df['Location'].value_counts()
    st.markdown("**Frekuensi Lokasi**")
    st.bar_chart(location)


with st.sidebar:
    st.header("Masukkan Informasi Anda")
    age = st.number_input("Usia (tahun)", min_value=0, max_value=100, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Non-Binary"])
    income = st.number_input("Pendapatan Tahunan (USD)", min_value=0, step=1000)
    travel_frequency = st.number_input("Frekuensi Liburan per Tahun", min_value=0, max_value=10, step=1)
    preferred_activity = st.selectbox("Aktivitas Favorit", ["Hiking", "Swimming", "Skiing", "Sunbathing"])
    location = st.selectbox("Tempat Tinggal", ["Urban", "Suburban", "Rural"])
    proximity_mountains = st.number_input("Jarak ke Gunung (miles)", min_value=0)
    proximity_beaches = st.number_input("Jarak ke Pantai (miles)", min_value=0)
    favorite_season = st.selectbox("Musim Favorit", ["Summer", "Winter", "Spring", "Fall"])
    pets = st.selectbox("Hewan Peliharaan?", ["Tidak", "Ya"])
    environmental_concerns = st.selectbox("Peduli Lingkungan?", ["Tidak", "Ya"])

# Konversi input menjadi numerik
gender_map = {"Female": 0, "Male": 1, "Non-Binary": 2}
preferred_activity_map = {"Hiking": 0, "Swimming": 1, "Skiing": 2, "Sunbathing": 3}
location_map = {"Urban": 0, "Suburban": 1, "Rural": 2}
favorite_season_map = {"Summer": 0, "Winter": 1, "Spring": 2, "Fall": 3}
pets_map = {"Tidak": 0, "Ya": 1}
environmental_concerns_map = {"Tidak": 0, "Ya": 1}

input_data = [
    age,
    income,
    travel_frequency,
    proximity_mountains,
    proximity_beaches,
    gender_map[gender],
    preferred_activity_map[preferred_activity],
    location_map[location],
    favorite_season_map[favorite_season],
    pets_map[pets],
    environmental_concerns_map[environmental_concerns],
]

input_data.extend([0, 0])  # Menambahkan kolom kosong jika diperlukan oleh model

# Pilih model untuk prediksi
model_choice = st.selectbox("Pilih Model", ["Random Forest", "XGBoost", "Feedforward Neural Network"])

# Membuat prediksi
if st.button('Prediksi'):
    prediction_proba = make_prediction(model_choice, input_data)
    result = 'Gunung' if prediction_proba[1] > prediction_proba[0] else 'Pantai'

    # Tampilkan hasil prediksi
    st.subheader(f"Prediksi Preferensi:")
    st.success(result)
    st.markdown(f"**Model yang digunakan**: {model_choice}")
    
    # Tampilkan probabilitas untuk Gunung dan Pantai
    st.markdown(f"**Probabilitas Gunung**: {prediction_proba[1]*100:.2f}%")
    st.markdown(f"**Probabilitas Pantai**: {prediction_proba[0]*100:.2f}%")

    # Menampilkan visualisasi probabilitas dengan progress bar
    data_df = pd.DataFrame(
        {
            "Preferensi": ["Gunung", "Pantai"],
            "Probabilitas": [f"{prediction_proba[1]*100:.2f}%", f"{prediction_proba[0]*100:.2f}%"],  # Probabilitas untuk Gunung dan Pantai
        }
    )

    # Tampilkan data editor dengan progress bar
    st.data_editor(
        data_df,
        column_config={
            "Probabilitas": st.column_config.ProgressColumn(
                "Probabilitas Prediksi",
                help="Probabilitas untuk memilih Gunung atau Pantai",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
        },
        hide_index=True,
    )
