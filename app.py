import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
@st.cache_resource
def load():
    movies = pd.read_csv("movies_final.csv")
    for c in ["genres", "keywords", "overview"]:
        movies[c] = movies[c].fillna("")
    movies["tags2"] = (movies["genres"] + " ") * 3 + (movies["keywords"] + " ") * 3 + movies["overview"]
    tfidf = TfidfVectorizer(stop_words="english", max_features=20000)
    matrix = tfidf.fit_transform(movies["tags2"])
    pop = np.log10(movies["vote_count"].values)
    pop = pop / pop.max()
    return movies, matrix, pop

movies, matrix, pop = load()

st.title("🎬 Movie Recommender")

title = st.selectbox("Pick a movie (type to search)", sorted(movies["title"]))
w = st.slider("Popularity weight", 0.0, 1.0, 0.5)

if st.button("Recommend"):
    i = movies.index[movies["title"] == title][0]
    scores = cosine_similarity(matrix[i], matrix).flatten()
    final = scores * ((1 - w) + w * pop)
    top = final.argsort()[::-1][1:6]
    result = pd.DataFrame({
        "Title": movies["title"].iloc[top].values,
        "Votes": movies["vote_count"].iloc[top].values,
        "Score": final[top].round(3),
    }) 
    st.table(result)
    
    