import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('kmeans_model.pkl')
# Load Scaler
scaler = joblib.load('scaler.pkl')
# Load the dataset
df = joblib.load("spotify.csv")
print(df.columns)
print(df.head())





# Cluster labels
cluster_names = {
    0: "Chill Vibes",
    1: "High Energy Dance",
    2: "Acoustic & Calm",
    3: "Moody & Emotional",
    4: "Upbeat Pop"
}
# Streamlit app

st.title("🎧 Streamify Playlist Generator")
st.write("Enter song features to discover its vibe and get recommendations.")

# User input for song features

danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
valence = st.slider("Valence (Mood)", 0.0, 1.0, 0.5)
tempo = st.slider("Tempo", 60, 200, 120)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.3)
speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
liveness = st.slider("Liveness", 0.0, 1.0, 0.1)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)

# Create a Dataframe and preprocess Use Inputs

if st.button("Generate Playlist"):
    song_features = pd.DataFrame([{
        "danceability": danceability,
        "energy": energy,
        "valence": valence,
        "tempo": tempo,
        "acousticness": acousticness,
        "speechiness": speechiness,
        "popularity": df["popularity"].mean(),
        "duration_ms": df["duration_ms"].mean(),
        "explicit": 0,
        "key": 5,
        "loudness": df["loudness"].mean(),
        "mode": 1,
        "instrumentalness": 0.0,
        "liveness": 0.1,
        "time_signature": 4
    }])
    
    # Select only numeric columns that the scaler was trained on (include bool for 'explicit')
    # Exclude 'cluster' which is the target, not a feature
    numeric_cols = df.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    if 'cluster' in numeric_cols:
        numeric_cols.remove('cluster')
    
    song_features = song_features.reindex(columns=numeric_cols, fill_value=0)

    song_scaled = scaler.transform(song_features)
    cluster_id = model.predict(song_scaled)[0]
    vibe = cluster_names[cluster_id]

    st.success(f"🎶 This song fits the **{vibe}** playlist")

# Get recommendations from the same cluster
    recommendations = (
        df[df["cluster"] == cluster_id]
        [["track_name", "artists", "track_genre"]]
        .sample(3)
    )

    st.subheader("Recommended Songs")
    st.dataframe(recommendations)


