import json
import streamlit as st
import pandas as pd  
from pycaret.clustering import load_model, predict_model  
import plotly.express as px  


MODEL_NAME = 'clustering_pipeline_v2.pkl'
DATA = 'welcome_survey_simple_v2.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v1.json'


@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def get_all_participants():
    model = get_model()
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)
    return df_with_clusters


# Sidebar: dane użytkownika
with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    age = st.selectbox("Wiek", ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

    person_df = pd.DataFrame([{
        'age': age,
        'edu_level': edu_level,
        'fav_animals': fav_animals,
        'fav_place': fav_place,
        'gender': gender,
    }])


# Główne dane
model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

# Przewidywanie klastra
predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]

# Obsługa formatu klucza
if isinstance(predicted_cluster_id, str) and predicted_cluster_id.startswith("Cluster "):
    cluster_key = predicted_cluster_id
else:
    cluster_key = f"Cluster {predicted_cluster_id}"

# Pobranie danych klastra
if cluster_key in cluster_names_and_descriptions:
    predicted_cluster_data = cluster_names_and_descriptions[cluster_key]
else:
    st.error(f"Nie znaleziono opisu dla klastra: {cluster_key}")
    st.stop()

# Wyświetlanie informacji o klastrze
st.header(f"Najbliżej Ci do grupy **{predicted_cluster_data['name']}**")
st.markdown(predicted_cluster_data['description'])

# Filtr osób z tego samego klastra
same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
st.metric("Liczba twoich znajomych", len(same_cluster_df))

# Wizualizacje
st.header("Osoby z grupy")

fig = px.histogram(same_cluster_df.sort_values("age"), x="age")
fig.update_layout(title="Rozkład wieku w grupie", xaxis_title="Wiek", yaxis_title="Liczba osób")
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="edu_level")
fig.update_layout(title="Rozkład wykształcenia w grupie", xaxis_title="Wykształcenie", yaxis_title="Liczba osób")
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_animals")
fig.update_layout(title="Rozkład ulubionych zwierząt w grupie", xaxis_title="Ulubione zwierzęta", yaxis_title="Liczba osób")
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_place")
fig.update_layout(title="Rozkład ulubionych miejsc w grupie", xaxis_title="Ulubione miejsce", yaxis_title="Liczba osób")
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="gender")
fig.update_layout(title="Rozkład płci w grupie", xaxis_title="Płeć", yaxis_title="Liczba osób")
st.plotly_chart(fig)