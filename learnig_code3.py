import os
import json
import pandas as pd
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import joblib
import time
from deep_translator import GoogleTranslator
from langdetect import detect
import tensorflow
# Отключаем предупреждения
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Отключает oneDNN предупреждения
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
# Определяем устройство
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Используемое устройство: {device}")
# Папка для сохранения модели
MODEL_DIR = "model2"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    """Загрузка и обработка данных."""
    print("[INFO] Загрузка данных...")
    movies = pd.read_csv('MovieSummaries/movie.metadata.tsv', sep='\t', header=None)
    movies.columns = ['movie_id', 'freebase_id', 'title', 'release_date', 'revenue', 'runtime', 'languages',
                      'countries', 'genres']
    # Парсим JSON-поля
    for col in ['genres', 'languages', 'countries']:
        movies[col] = movies[col].apply(lambda x: list(json.loads(x).values()) if pd.notna(x) else [])
    plots = pd.read_csv('MovieSummaries/plot_summaries.txt', sep='\t', header=None)
    plots.columns = ['movie_id', 'plot']
    movies = movies.merge(plots, on='movie_id')
    print(f"[INFO] Загружено {len(movies)} фильмов.")
    return movies


def vectorize_text(movies):
    """Векторизация сюжетов фильмов."""
    print("[INFO] Векторизация сюжетов...")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=device)
    movies['plot_embedding'] = movies['plot'].apply(lambda x: model.encode(str(x), device=device))
    return movies, model


def train_knn(movies):
    """Обучение KNN модели."""
    print("[INFO] Обучение KNN модели...")
    X = np.array(movies['plot_embedding'].tolist())
    knn = NearestNeighbors(n_neighbors=50, metric='cosine')
    knn.fit(X)

    # Сохраняем KNN модель
    joblib.dump(knn, os.path.join(MODEL_DIR, "knn_model.pkl"))
    print(f"[INFO] KNN модель сохранена в {MODEL_DIR}/knn_model.pkl")
    return knn


def save_embeddings(movies):
    """Сохранение векторизованных данных."""
    embeddings_path = os.path.join(MODEL_DIR, "movie_embeddings.npy")
    metadata_path = os.path.join(MODEL_DIR, "movies_metadata.pkl")

    np.save(embeddings_path, np.array(movies['plot_embedding'].tolist()))
    movies.drop(columns=['plot_embedding']).to_pickle(metadata_path)

    print(f"[INFO] Векторные представления сохранены в {embeddings_path}")
    print(f"[INFO] Метаданные фильмов сохранены в {metadata_path}")


def recommend_movies(query_text, model, knn, movies, top_n=5):
    """Рекомендация фильмов на основе текста запроса."""
    query_embed = model.encode(query_text, device=device)
    distances, indices = knn.kneighbors([query_embed])
    recommendations = movies.iloc[indices[0]]
    return recommendations.head(top_n)


def recommend_from_history(watched_ids, knn, movies, top_n=5):
    """Рекомендация фильмов на основе истории просмотров."""
    valid_watched_ids = [mid for mid in watched_ids if mid in movies['movie_id'].values]
    if not valid_watched_ids:
        print("[WARN] Нет данных для просмотренных фильмов.")
        return pd.DataFrame()

    watched_embeddings = movies[movies['movie_id'].isin(valid_watched_ids)]['plot_embedding']
    avg_embedding = np.mean(watched_embeddings.tolist(), axis=0)
    distances, indices = knn.kneighbors([avg_embedding])
    return movies.iloc[indices[0]].head(top_n)


if __name__ == "__main__":
    total_start_time = time.time()

    # Проверяем, существует ли обученная модель
    knn_path = os.path.join(MODEL_DIR, "knn_model.pkl")
    embeddings_path = os.path.join(MODEL_DIR, "movie_embeddings.npy")
    metadata_path = os.path.join(MODEL_DIR, "movies_metadata.pkl")

    if os.path.exists(knn_path) and os.path.exists(embeddings_path) and os.path.exists(metadata_path):
        print("[INFO] Найдена сохранённая модель, загружаем...")
        knn = joblib.load(knn_path)
        movies = pd.read_pickle(metadata_path)
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=device)
        print("[INFO] Модель загружена успешно.")
    else:
        print("[INFO] Обученная модель не найдена, начинаем обучение...")
        movies = load_data()
        movies, model = vectorize_text(movies)
        knn = train_knn(movies)
        save_embeddings(movies)

    total_elapsed_time = time.time() - total_start_time
    print(f"[INFO] Полное время выполнения: {total_elapsed_time:.2f} сек.")


    def translate_if_needed(text, target_lang="en"):
        """Переводит текст на английский, если он не на английском."""
        detected_lang = detect(text)
        return GoogleTranslator(source=detected_lang, target=target_lang).translate(
            text) if detected_lang != "en" else text


    # Примеры рекомендаций
    query = "История о любви и войне в послевоенной Германии"  # Можно вводить и на русском, и на английском
    translated_query = translate_if_needed(query)
    print(f"[INFO] Запрос после перевода: {translated_query}")

    print("[INFO] Рекомендации по запросу:", query)
    print(recommend_movies(translated_query, model, knn, movies)[['title', 'genres']])

    watched_ids = [18496109, 34689049, 9916335,20904045]
    print("[INFO] Рекомендации на основе истории просмотров:")
    recommendations = recommend_from_history(watched_ids, knn, movies)
    if not recommendations.empty:
        print(recommendations[['title', 'genres']])
    else:
        print("Нет данных для рекомендаций.")