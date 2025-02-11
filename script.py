import os
import gzip
import shutil
import urllib.request

# Список файлов IMDb
files = [
    "title.basics.tsv.gz",
    "title.ratings.tsv.gz",
    "name.basics.tsv.gz",
    "title.crew.tsv.gz",
    "title.episode.tsv.gz",
    "title.principals.tsv.gz",
    "title.akas.tsv.gz"
]

# Папка для сохранения данных
data_dir = "imdb_data"
os.makedirs(data_dir, exist_ok=True)

# Функция для скачивания и разархивирования
def download_and_extract(file_name):
    url = f"https://datasets.imdbws.com/{file_name}"
    gz_path = os.path.join(data_dir, file_name)
    tsv_path = gz_path.replace(".gz", "")

    print(f"Скачивание {file_name}...")
    urllib.request.urlretrieve(url, gz_path)

    print(f"Распаковка {file_name}...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(tsv_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(gz_path)  # Удаляем архив после распаковки
    print(f"Файл сохранен: {tsv_path}")

# Скачиваем все файлы
for file in files:
    download_and_extract(file)

print("✅ Все файлы скачаны и распакованы!")