import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from scipy.sparse import load_npz

# Директория с моделями
MODEL_DIR = 'models'


# Проверка наличия файлов модели
def check_model_files():
    files = [
        os.path.join(MODEL_DIR, 'word2vec.model'),
        os.path.join(MODEL_DIR, 'tfidf.pkl'),
        os.path.join(MODEL_DIR, 'tfidf_matrix.npz'),
        os.path.join(MODEL_DIR, 'movies_metadata.csv')
    ]
    missing_files = [file for file in files if not os.path.exists(file)]
    if missing_files:
        raise FileNotFoundError(f"Missing model files: {missing_files}")
    print("All model files are present.")


# Загрузка моделей и данных
def load_models():
    check_model_files()
    print("Loading models...")
    w2v_model = Word2Vec.load(os.path.join(MODEL_DIR, 'word2vec.model'))
    tfidf = pd.read_pickle(os.path.join(MODEL_DIR, 'tfidf.pkl'))
    tfidf_matrix = load_npz(os.path.join(MODEL_DIR, 'tfidf_matrix.npz'))
    movies = pd.read_csv(os.path.join(MODEL_DIR, 'movies_metadata.csv'))
    print("Models loaded successfully.")
    return w2v_model, tfidf, tfidf_matrix, movies


# Проверка модели Word2Vec
def validate_word2vec(w2v_model):
    test_words = ['drama', 'comedy', 'action']
    for word in test_words:
        if word in w2v_model.wv:
            print(f"Top similar words to '{word}':", w2v_model.wv.most_similar(word, topn=5))
        else:
            print(f"Word '{word}' not found in Word2Vec model.")


# Проверка модели TF-IDF
def validate_tfidf(tfidf, movies):
    sample_text = movies['tags'].sample(1).values[0]
    tfidf_vector = tfidf.transform([sample_text])
    print("TF-IDF vector shape:", tfidf_vector.shape)
    print("Sample TF-IDF features:", tfidf.get_feature_names_out()[:10])


# Проверка рекомендаций на основе TF-IDF
def test_recommendations(tfidf_matrix, movies, title, top_n=5):
    try:
        idx = movies[movies['primaryTitle'].str.lower() == title.lower()].index[0]
    except IndexError:
        print(f"Movie '{title}' not found.")
        return

    target_vector = tfidf_matrix[idx]
    sim_scores = cosine_similarity(target_vector, tfidf_matrix).flatten()
    sim_indices = np.argsort(sim_scores)[::-1][1:top_n + 1]
    print(f"Top {top_n} recommendations for '{title}':")
    print(movies.iloc[sim_indices][['primaryTitle', 'tags']])


if __name__ == '__main__':
    w2v_model, tfidf, tfidf_matrix, movies = load_models()
    validate_word2vec(w2v_model)
    validate_tfidf(tfidf, movies)
    test_recommendations(tfidf_matrix, movies, 'The Matrix')
