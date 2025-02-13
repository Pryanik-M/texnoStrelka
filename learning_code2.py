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
from scipy.sparse import save_npz, load_npz
from typing import List, Dict

# Конфигурация
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)
DATA_DIR = "C:/Users/utu/PycharmProjects/Mashine_learns/imdb_data"


def load_data() -> pd.DataFrame:
    """Загрузка и объединение данных с оптимизацией типов данных"""
    required_files = np.array(['title.basics.tsv', 'title.ratings.tsv', 'title.crew.tsv'])
    file_paths = np.char.add(DATA_DIR + '/', required_files)

    if not np.all(np.vectorize(os.path.exists)(file_paths)):
        missing_files = required_files[~np.vectorize(os.path.exists)(file_paths)]
        raise FileNotFoundError(f"Отсутствуют файлы: {', '.join(missing_files)}")

    dtypes_basics = {
        'tconst': 'category',
        'titleType': 'category',
        'primaryTitle': 'str',
        'isAdult': 'float32',
        'genres': 'str',
        'startYear': 'str'
    }

    title_basics = pd.read_csv(
        os.path.join(DATA_DIR, 'title.basics.tsv'),
        sep='\t',
        dtype=dtypes_basics,
        na_values=['\\N'],
        usecols=list(dtypes_basics.keys())
    )

    title_basics['isAdult'] = np.where(title_basics['isAdult'].isna(), 0, title_basics['isAdult']).astype('int8')

    title_ratings = pd.read_csv(
        os.path.join(DATA_DIR, 'title.ratings.tsv'),
        sep='\t',
        dtype={'tconst': 'category', 'averageRating': 'float32', 'numVotes': 'int32'}
    )

    title_crew = pd.read_csv(
        os.path.join(DATA_DIR, 'title.crew.tsv'),
        sep='\t',
        dtype={'tconst': 'category', 'directors': 'str', 'writers': 'str'}
    )

    movies = title_basics.merge(title_ratings, on='tconst').merge(title_crew, on='tconst')

    mask = (
            movies['titleType'].isin(['movie', 'tvSeries']) &
            (movies['numVotes'] > 1000) &
            movies['startYear'].str.isnumeric()
    )

    return movies[mask].reset_index(drop=True)


class TextPreprocessor:
    """Оптимизированный препроцессинг текста с кэшированием стоп-слов"""

    def __init__(self):
        self.stemmer = SnowballStemmer('english')
        self.stop_words = frozenset(stopwords.words('english'))
        self.regex = re.compile(r'[^a-z0-9\s]')
        self.nm_filter = re.compile(r'^nm\d+')

    def preprocess(self, items):
        """Векторизованная обработка списка тегов"""
        items = np.array(items, dtype=str)
        mask = ~pd.isna(items) & ~np.vectorize(lambda x: bool(self.nm_filter.match(x)))(items)
        items = np.char.lower(items[mask])
        items = np.char.replace(np.vectorize(lambda x: self.regex.sub('', x))(items), '  ', ' ')
        tokens = np.char.split(items)
        tokens = np.concatenate(tokens)
        tokens = np.array([self.stemmer.stem(word) for word in tokens if word not in self.stop_words and len(word) > 2])
        return tokens.tolist()


def generate_tags(movies: pd.DataFrame) -> pd.DataFrame:
    """Генерация тегов с фильтрацией nm-идентификаторов"""
    principals = pd.read_csv(
        os.path.join(DATA_DIR, 'title.principals.tsv'),
        sep='\t',
        usecols=['tconst', 'category', 'characters'],
        dtype={'tconst': 'category', 'category': 'category', 'characters': 'str'}
    )

    # Фильтрация только актеров и актрис
    characters = principals.loc[principals['category'].isin(['actor', 'actress'])]
    characters = characters.groupby('tconst', observed=True)['characters'].agg(
        lambda x: list(x.dropna().astype(str).str.replace(r'[\[\]"]', '', regex=True)))

    # Объединение с movies
    movies = movies.join(characters.rename('characters'), on='tconst', how='left')

    # Векторизированная обработка полей
    tag_columns = {
        'genres': movies['genres'].str.split(',').fillna('').apply(lambda x: x if isinstance(x, list) else []),
        'directors': movies['directors'].str.split(',').fillna('').apply(lambda x: x if isinstance(x, list) else []),
        'writers': movies['writers'].str.split(',').fillna('').apply(lambda x: x if isinstance(x, list) else []),
        'characters': movies['characters'].fillna('').apply(lambda x: x if isinstance(x, list) else []),
        'year': movies['startYear'].apply(lambda x: [x] if pd.notna(x) else []),
        'rating': movies['averageRating'].apply(lambda x: [f"rating_{int(x)}"] if pd.notna(x) else [])
    }

    # Фильтрация nm-идентификаторов (используем apply вместо np.vectorize)
    for key in ['directors', 'writers']:
        tag_columns[key] = tag_columns[key].apply(lambda tags: [t for t in tags if not t.startswith('nm')])

    # Создание колонки tags
    movies['tags'] = pd.Series([{k: tag_columns[k].iloc[i] for k in tag_columns.keys()} for i in range(len(movies))])

    return movies


def train_models(movies: pd.DataFrame) -> tuple:
    """Обучение моделей с использованием numpy для обработки"""
    preprocessor = TextPreprocessor()

    # Векторизованная подготовка данных для Word2Vec
    tagged_data = np.array([
        preprocessor.preprocess(
            [str(tag) for category in row.values() for tag in category]
        )
        for row in movies['tags']
    ], dtype=object)

    # Обучение Word2Vec с оптимизированными параметрами
    w2v_model = Word2Vec(
        sentences=tagged_data,
        vector_size=256,
        window=7,
        min_count=2,
        workers=multiprocessing.cpu_count(),
        epochs=15,
        hs=1  # Использование hierarchical softmax для улучшения качества
    )

    # TF-IDF с оптимизированными параметрами
    tfidf = TfidfVectorizer(
        max_features=5000,
        dtype=np.float32,
        sublinear_tf=True,  # Улучшение весов
        analyzer='word',
        ngram_range=(1, 2)  # Учет биграмм
    )
    tfidf_matrix = tfidf.fit_transform(
        [' '.join(map(str, tags)) for tags in tagged_data]
    )

    return w2v_model, tfidf, tfidf_matrix


class MovieRecommender:
    """Оптимизированная система рекомендаций"""

    def __init__(self):
        self.metadata = pd.read_csv(
            os.path.join(MODEL_DIR, 'movies_metadata.csv'),
            dtype={'tconst': 'category', 'primaryTitle': 'str'}
        )
        self.tfidf = pd.read_pickle(os.path.join(MODEL_DIR, 'tfidf.pkl'))
        self.w2v_model = Word2Vec.load(os.path.join(MODEL_DIR, 'word2vec.model'))
        self.tfidf_matrix = load_npz(os.path.join(MODEL_DIR, 'tfidf_matrix.npz'))

        # Предварительное вычисление норм векторов
        self.tfidf_norms = np.sqrt(np.asarray(self.tfidf_matrix.power(2).sum(axis=1)).ravel())

    def get_similar_movies(self, title: str, top_n: int = 10) -> pd.DataFrame:
        """Оптимизированный поиск с использованием предвычисленных норм"""
        try:
            idx = self.metadata[self.metadata['primaryTitle'].str.lower() == title.lower()].index[0]
        except IndexError:
            return pd.DataFrame()  # Возвращаем пустой DataFrame вместо строки

        # Оптимизированное вычисление косинусной схожести
        target_vector = self.tfidf_matrix[idx]
        dot_product = self.tfidf_matrix.dot(target_vector.T).toarray().ravel()
        sim_scores = dot_product / (self.tfidf_norms * self.tfidf_norms[idx] + 1e-9)

        # Использование argpartition для оптимизации выбора топ-N
        sim_indices = np.argpartition(-sim_scores, top_n)[:top_n]
        return self.metadata.iloc[sim_indices][['primaryTitle', 'tags']]


# Сохранение моделей
def save_models(w2v_model, tfidf, movies, tfidf_matrix):
    w2v_model.save(os.path.join(MODEL_DIR, 'word2vec.model'))
    pd.to_pickle(tfidf, os.path.join(MODEL_DIR, 'tfidf.pkl'))
    save_npz(os.path.join(MODEL_DIR, 'tfidf_matrix.npz'), tfidf_matrix)  # Сохраняем разреженную матрицу
    movies[['tconst', 'primaryTitle', 'tags']].to_csv(
        os.path.join(MODEL_DIR, 'movies_metadata.csv'), index=False)


def main():
    # Оптимизированная загрузка моделей
    model_files = ['word2vec.model', 'tfidf.pkl', 'movies_metadata.csv']
    if all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in model_files):
        print("Использование предварительно обученных моделей")
        return

    print("Загрузка данных...")
    movies = load_data()
    print("Генерация тегов...")
    movies = generate_tags(movies)
    print("Обучение моделей...")
    w2v_model, tfidf, tfidf_matrix = train_models(movies)
    print("Сохранение моделей...")
    save_models(w2v_model, tfidf, movies, tfidf_matrix)


if __name__ == '__main__':
    main()
