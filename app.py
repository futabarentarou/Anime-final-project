import streamlit as st
import pandas as pd
import joblib

model = joblib.load("anime_model.pkl")
model_columns = joblib.load("model_columns.pkl")
best_threshold = joblib.load("best_threshold.pkl")

st.title("Anime High Score Predictor")

st.write("Predice si un anime podría tener una puntuación alta.")

num_list_users = st.number_input("Usuarios que agregaron el anime a su lista", min_value=0, value=10000)
num_scoring_users = st.number_input("Usuarios que calificaron el anime", min_value=0, value=5000)
num_episodes = st.number_input("Número de episodios", min_value=0, value=12)
average_episode_duration = st.number_input("Duración promedio del episodio en segundos", min_value=0, value=1440)

media_type = st.selectbox("Tipo de anime", ["tv", "movie", "ova", "ona", "special", "music"])
rating = st.selectbox("Rating", ["g", "pg", "pg_13", "r", "r+", "rx"])

input_data = pd.DataFrame({
    "num_list_users": [num_list_users],
    "num_scoring_users": [num_scoring_users],
    "num_episodes": [num_episodes],
    "average_episode_duration": [average_episode_duration],
    "media_type": [media_type],
    "rating": [rating]
})

input_encoded = pd.get_dummies(input_data)
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

if st.button("Predecir"):
    probability = model.predict_proba(input_encoded)[0][1]
    prediction = int(probability >= best_threshold)

    st.write(f"Probabilidad de alta puntuación: {probability:.2%}")

    if prediction == 1:
        st.success("Predicción: Anime con alta puntuación")
    else:
        st.warning("Predicción: Anime sin alta puntuación")