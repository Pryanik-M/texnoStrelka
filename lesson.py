import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import multiprocessing
from scipy.sparse import save_npz, load_npz  # Для работы с разреженными матрицами

# Конфигурация
DATA_DIR = 'imdb_data'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)


# Загрузка и обработка данных
def load_data():
    # Загрузка данных с использованием float для столбца 'isAdult', чтобы поддерживать NaN
    title_basics = pd.read_csv(
        os.path.join(DATA_DIR, 'title.basics.tsv'),
        sep='\t',
        dtype={'tconst': 'str', 'titleType': 'str', 'primaryTitle': 'str',
               'originalTitle': 'str', 'isAdult': 'float',  # Используем float для поддержки NaN
               'startYear': 'str', 'endYear': 'str', 'runtimeMinutes': 'str', 'genres': 'str'},
        na_values=['\\N']
    )

    # Заменяем пропуски в столбце 'isAdult' на 0 (или False)
    title_basics['isAdult'] = title_basics['isAdult'].fillna(0).astype(int)

    # Загрузка рейтингов
    title_ratings = pd.read_csv(
        os.path.join(DATA_DIR, 'title.ratings.tsv'),
        sep='\t',
        dtype={'tconst': 'str', 'averageRating': 'float32', 'numVotes': 'int32'}
    )

    # Загрузка информации о съемочной группе
    title_crew = pd.read_csv(
        os.path.join(DATA_DIR, 'title.crew.tsv'),
        sep='\t',
        dtype={'tconst': 'str', 'directors': 'str', 'writers': 'str'}
    )

    # Объединение данных
    movies = title_basics.merge(title_ratings, on='tconst', how='inner')  # Используем inner join для фильтрации
    movies = movies.merge(title_crew, on='tconst', how='left')

    # Фильтрация только фильмов и сериалов с достаточным количеством оценок
    movies = movies[
        (movies['titleType'].isin(['movie', 'tvSeries'])) &
        (movies['numVotes'] > 1000)  # Фильтр по количеству голосов
        ]

    return movies


# Генерация тегов для фильмов
def generate_tags(movies):
    # Загрузка дополнительных данных
    principals = pd.read_csv(
        os.path.join(DATA_DIR, 'title.principals.tsv'),
        sep='\t',
        usecols=['tconst', 'category', 'characters'],
        dtype={'tconst': 'str', 'category': 'str', 'characters': 'str'}
    )

    # Обработка жанров
    movies['genres'] = movies['genres'].str.split(',')

    # Извлечение ключевых персонажей
    characters = principals[principals['category'].isin(['actor', 'actress'])]
    characters = characters.groupby('tconst')['characters'].apply(
        lambda x: [re.sub(r'[\[\]"]', '', c) for c in x.dropna()]
    ).reset_index()

    # Объединение с основными данными
    movies = movies.merge(characters, on='tconst', how='left')

    # Создание тегов
    movies['tags'] = movies.apply(lambda row: {
        'genres': row['genres'] if isinstance(row['genres'], list) else [],
        'directors': row['directors'].split(',') if pd.notna(row['directors']) else [],
        'writers': row['writers'].split(',') if pd.notna(row['writers']) else [],
        'characters': row['characters'] if isinstance(row['characters'], list) else [],
        'year': [row['startYear']] if pd.notna(row['startYear']) else [],
        'rating': [f"rating_{int(row['averageRating'])}"] if pd.notna(row['averageRating']) else []
    }, axis=1)

    return movies


# Предобработка текста
class TextPreprocessor:
    def __init__(self):
        self.stemmer = SnowballStemmer('english')
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, text):
        if isinstance(text, list):
            return [self.preprocess_item(item) for item in text]
        return self.preprocess_item(text)

    def preprocess_item(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        tokens = [self.stemmer.stem(word) for word in text.split()
                  if word not in self.stop_words and len(word) > 2]
        return ' '.join(tokens)


# Обучение моделей
def train_models(movies):
    # Подготовка данных для Word2Vec
    tagged_data = []
    for _, row in movies.iterrows():
        tags = []
        tags.extend(row['tags']['genres'])
        tags.extend(row['tags']['directors'])
        tags.extend(row['tags']['writers'])
        tags.extend(row['tags']['characters'])
        tags.extend(row['tags']['year'])
        tags.extend(row['tags']['rating'])
        tagged_data.append([tag.lower().replace(' ', '_') for tag in tags])

    # Обучение Word2Vec модели
    w2v_model = Word2Vec(
        sentences=tagged_data,
        vector_size=128,  # Уменьшаем размер вектора
        window=5,
        min_count=1,
        workers=multiprocessing.cpu_count(),
        epochs=10
    )

    # Создание TF-IDF векторов
    preprocessor = TextPreprocessor()
    movies['processed_tags'] = movies['tags'].apply(
        lambda x: ' '.join([preprocessor.preprocess_item(tag)
                            for category in x.values()
                            for tag in (category if isinstance(category, list) else [category])])
    )

    # Уменьшаем количество фичей и используем разреженный формат
    tfidf = TfidfVectorizer(max_features=2000, dtype=np.float32)
    tfidf_matrix = tfidf.fit_transform(movies['processed_tags'])

    return w2v_model, tfidf, tfidf_matrix  # Не возвращаем cosine_sim


# Сохранение моделей
def save_models(w2v_model, tfidf, movies, tfidf_matrix):
    w2v_model.save(os.path.join(MODEL_DIR, 'word2vec.model'))
    pd.to_pickle(tfidf, os.path.join(MODEL_DIR, 'tfidf.pkl'))
    save_npz(os.path.join(MODEL_DIR, 'tfidf_matrix.npz'), tfidf_matrix)  # Сохраняем разреженную матрицу
    movies[['tconst', 'primaryTitle', 'tags']].to_csv(
        os.path.join(MODEL_DIR, 'movies_metadata.csv'), index=False)


def main():
    print("Loading data...")

    # Проверим, есть ли уже сохраненные модели
    if os.path.exists(os.path.join(MODEL_DIR, 'word2vec.model')) and \
            os.path.exists(os.path.join(MODEL_DIR, 'tfidf.pkl')):
        print("Models already exist, skipping training.")
        # Загрузим уже существующие модели
        w2v_model = Word2Vec.load(os.path.join(MODEL_DIR, 'word2vec.model'))
        tfidf = pd.read_pickle(os.path.join(MODEL_DIR, 'tfidf.pkl'))
        movies = pd.read_csv(os.path.join(MODEL_DIR, 'movies_metadata.csv'))
    else:
        movies = load_data()
        print("Generating tags...")
        movies = generate_tags(movies)
        print("Training models...")
        w2v_model, tfidf, tfidf_matrix = train_models(movies)
        print("Saving models...")
        save_models(w2v_model, tfidf, movies, tfidf_matrix)
        print("Training completed!")


# Класс рекомендаций
class MovieRecommender:
    def __init__(self):
        self.movies = pd.read_csv(os.path.join(MODEL_DIR, 'movies_metadata.csv'))
        self.tfidf = pd.read_pickle(os.path.join(MODEL_DIR, 'tfidf.pkl'))
        self.w2v_model = Word2Vec.load(os.path.join(MODEL_DIR, 'word2vec.model'))
        self.tfidf_matrix = load_npz(os.path.join(MODEL_DIR, 'tfidf_matrix.npz'))  # Загружаем разреженную матрицу

    def get_similar_movies(self, title, top_n=10):
        try:
            idx = self.movies[self.movies['primaryTitle'].str.lower() == title.lower()].index[0]
        except IndexError:
            return f"Movie '{title}' not found"

        # Вычисляем схожести только для нужного фильма
        target_vector = self.tfidf_matrix[idx]
        sim_scores = cosine_similarity(target_vector, self.tfidf_matrix).flatten()
        sim_indices = np.argpartition(sim_scores, -top_n - 1)[-top_n - 1:-1]
        return self.movies.iloc[sim_indices][['primaryTitle', 'tags']]

    def recommend_by_tags(self, user_tags, top_n=10):
        preprocessor = TextPreprocessor()
        processed_tags = preprocessor.preprocess(user_tags)
        tfidf_vector = self.tfidf.transform([' '.join(processed_tags)])
        sim_scores = cosine_similarity(tfidf_vector, self.tfidf_matrix)
        sim_indices = np.argsort(sim_scores[0])[::-1][:top_n]
        return self.movies.iloc[sim_indices][['primaryTitle', 'tags']]

    def expand_tags(self, tags):
        expanded = set(tags)
        for tag in tags:
            if tag in self.w2v_model.wv:
                similar = self.w2v_model.wv.most_similar(tag, topn=3)
                expanded.update([t[0] for t in similar])
        return list(expanded)


if __name__ == '__main__':
    main()