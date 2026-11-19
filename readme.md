# 🎬 Movie Recommender

Pick a movie and get 5 similar ones. built with Python, scikit-learn and Streamlit. It runs fully offline on a 4GB RAM laptop.

SCREENSHOT:-
screenshot.png

## How it works
1. **Text to numbers:** each movie's genres, keywords and overview are combined into one text and turned into word weights with TF-IDF (rare, meaningful words like "heist" count more than common ones). Genres and keywords are repeated 3 times so they count more than the long plot text.
2. **Finding similar movies:** cosine similarity compares the picked movie with all 17,290 others, one row at a time, which keeps memory small.
3. **Popularity blend:** the score is multiplied by `(1 - w) + w * pop`, where `pop` is the log of the vote count scaled to 0-1. This pushes well-known movies above obscure ones that happen to share words. The slider sets `w`.
4. **No self-match:** the picked movie's own score is set to 0 so it never recommends itself.

## Dataset
[TMDB Movies Dataset](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies) by asaniczka on Kaggle. 
I kept movies that have 100+ votes, a genre and an overview, and removed duplicate titles (keeping the most-voted one). That left 17,290 movies. The raw file (about 645 MB) is not in this repo.

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitations
- It matches on words, not meaning, so sequels with different plot words can rank low.
- Next step: replace TF-IDF with sentence embeddings.