from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_required, current_user, logout_user, login_user
import os
import smtplib
from email.mime.text import MIMEText
from random import randint
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import os
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import joblib
from deep_translator import GoogleTranslator
from langdetect import detect
from Levenshtein import distance as levenshtein_distance

app = Flask(__name__, static_folder='static')


app.config['SECRET_KEY'] = 'hardsecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///dbase.db'
# app.config['STATIC'] = 'static'
db = SQLAlchemy(app)

IMAGES_FOLDER = os.path.join('static', 'img')
app.config['UPLOAD_FOLDER'] = IMAGES_FOLDER

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_DIR = "model2"
os.makedirs(MODEL_DIR, exist_ok=True)

CODE = 0


class User(UserMixin, db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(256))  # Увеличиваем длину для хранения хэша
    name = db.Column(db.String(100))
    surname = db.Column(db.String(100))
    age = db.Column(db.String(100))

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    movie_name = db.Column(db.String(100))
    text = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    user = db.relationship('User', backref='comments')



with app.app_context():
    db.create_all()


def recommend_movies(query_text, model, knn, movies, top_n=5):
    query_embed = model.encode(query_text, device=device)
    distances, indices = knn.kneighbors([query_embed])
    recommendations = movies.iloc[indices[0]]
    if top_n is None:
        return recommendations
    else:
        return recommendations.head(top_n)


def search_movies_by_title(query, movies, top_n=5):
    translated_query = translate_if_needed(query)
    mask = movies['title'].str.contains(translated_query, case=False, na=False)
    results = movies[mask].head(top_n)
    return results


def translate_if_needed(text, target_lang="en"):
    try:
        detected_lang = detect(text)
    except Exception:
        detected_lang = "en"
    return GoogleTranslator(source=detected_lang, target=target_lang).translate(
        text) if detected_lang != "en" else text


def send_email(message, adress):
    sender = "gulovskiu@gmail.com"
    password = "nwjcfhzloyluetwv"
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    try:
        server.login(sender, password)
        msg = MIMEText(message)
        msg["Subject"] = "Подтверждение почты"
        server.sendmail(sender, adress, msg.as_string())
        return "The message was sent successfully!"
    except Exception as _ex:
        return f"{_ex}\nCheck your login or password please!"


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if current_user.is_authenticated:
            movie_name = request.form.get('movie_name')
            comment_text = request.form.get('comment_text')
            if comment_text and movie_name:
                new_comment = Comment(
                    user_id=current_user.id,
                    movie_name=movie_name,
                    text=comment_text
                )
                db.session.add(new_comment)
                db.session.commit()
            return redirect(url_for('index'))
        else:
            return redirect(url_for('login'))
    comments = Comment.query.order_by(Comment.timestamp.desc()).all()
    flag_user = current_user.is_authenticated
    return render_template('index.html', flag=flag_user, comments=comments)


@app.route('/new_password', methods=['GET', 'POST'])
def new_password():
    email = request.args.get('email')
    if request.method == 'POST':
        psw1 = request.form['psw1']
        psw2 = request.form['psw2']
        if not psw1 or not psw2:
            return render_template('new_password.html', err='Заполните все поля', psw1=psw1, psw2=psw2)
        if len(psw1) < 8:
            return render_template('new_password.html', err='Пароль слишком маленький', psw1=psw1, psw2=psw2)
        if psw1 != psw2:
            return render_template('new_password.html', err='Пароли различаются', psw1=psw1, psw2=psw2)
        user = User.query.filter_by(email=email).first()
        if user:
            user.password = generate_password_hash(psw1)
            db.session.commit()
        return redirect(url_for('login'))
    else:
        return render_template('new_password.html', psw1='', psw2='')


@app.route('/send', methods=['GET', 'POST'])
def send():
    email = request.args.get('email')
    if request.method == 'POST':
        global CODE
        reset = int(request.args.get('reset'))
        email = request.form['mail']
        unic_code = request.form['unik_cod']
        if email == '':
            return render_template('password.html', flag=False, err="Введите почту")
        if unic_code == '':
            CODE = randint(1000, 9999)
            message = (f'''Здравствуйте!
            Вы получили это письмо, потому что мы получили запрос на подтверждения почты для вашей учетной записи.
            Специальный код: {CODE}
            Если вы не запрашивали код, никаких дальнейших действий не требуется.

            С Уважением,
            KinoStrelka платформа''')
            send_email(message=message, adress=email)
            return render_template('password.html', flag=True, err="Код отправлен", email=email)
        else:
            if int(unic_code) == CODE:
                CODE = 0
                session['email'] = email
                if reset == 1:
                    return redirect(url_for('new_password', email=email))
                elif reset == 0:
                    return redirect(url_for('index'))
                else:
                    return redirect(url_for('send', email=email))
    else:
        return render_template('password.html', flag=False, err="", email=email)


@app.route('/profile')
def profile():
    return render_template('profile.html', name=current_user.name, surname=current_user.surname,
                           age=current_user.age, email=current_user.email, image='static/image/profile_rev.png')

@app.route('/movie')
def movie():
    video_path = request.args.get('name', 'videos/primer.mp4')
    return render_template('movie.html', name=video_path)


@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    email = request.args.get('email')
    user = User.query.filter_by(email=email).first()
    if request.method == 'POST':
        if request.form.get('reset'):
            return redirect(url_for('new_password', email=email))
        new_name = request.form['name']
        new_surname = request.form['surname']
        new_age = request.form['age']
        if not new_age or not new_surname or not new_name:
            return render_template('create_profile.html', err='Заполнены не все поля',
                                   name=new_name, surname=new_surname, age=new_age)
        if user != '':
            user.name = new_name
            user.surname = new_surname
            user.age = new_age
            db.session.commit()
            return redirect(url_for('profile'))
    else:
        return render_template('create_profile.html', surname=user.surname, name=user.name,
                               age=user.age, email=email)


@app.route('/entrance', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('mail')
        password = request.form.get('password')
        remember = request.form.get('rememder')
        if remember:
            remember = True
        else:
            remember = False
        if not email or not password:
            return render_template('entrance.html', err='Заполнены не все поля', email=email,
                                   password=password, remember=remember)
        if '@' not in email:
            return render_template('entrance.html', err='Неправильно указана почта', email=email,
                                   password=password, remember=remember)
        if len(password) < 8:
            return render_template('entrance.html', err='Слабый пароль', email=email,
                                   password='', remember=remember)
        user = User.query.filter_by(email=email).first()
        if not user:
            return render_template('entrance.html', err='Почта не зарегистрирована', email=email,
                                   password=password, remember=remember)
        user = User.query.filter_by(email=email).first()
        if not user:
            return render_template('entrance.html', err='Почта не зарегистрирована', email=email,
                                   password=password, remember=remember)
        if not check_password_hash(user.password, password):
            return render_template('entrance.html', err='Проверьте данные', email=email,
                                   password=password, remember=remember)
        login_user(user, remember=remember)
        return redirect(url_for('index'))
    else:
        return render_template('entrance.html', err='')


@app.route('/recomendations', methods=['GET', 'POST'])
def recomendations():
    results = None
    if request.method == 'POST':
        query = request.form.get('query')
        if query:
            knn_path = os.path.join(MODEL_DIR, "knn_model.pkl")
            embeddings_path = os.path.join(MODEL_DIR, "movie_embeddings.npy")
            metadata_path = os.path.join(MODEL_DIR, "movies_metadata.pkl")
            if os.path.exists(knn_path) and os.path.exists(embeddings_path) and os.path.exists(metadata_path):
                knn = joblib.load(knn_path)
                movies = pd.read_pickle(metadata_path)
                model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=device)
                translated_query = translate_if_needed(query)
                print(f"[INFO] Запрос после перевода: {translated_query}")
                recommendations = recommend_movies(translated_query, model, knn, movies, top_n=None)
                results = recommendations[['title']].to_dict('records')
    return render_template('recomendations.html', results=results)


@app.route('/signup', methods=['POST', 'GET'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        name = request.form.get('name')
        surname = request.form.get('surname')
        password = request.form.get('password')
        password1 = request.form.get('password1')
        age = request.form.get('age')
        if not email or not name or not surname or not password or not password1 or not age:
            return render_template('n_3.html', err="Заполнены не все поля", email=email,
                                   name=name, surname=surname, password=password, password1=password1, age=age)
        if '@' not in email:
            return render_template('n_3.html', err="Некорректная почта", email=email,
                                   name=name, surname=surname, password=password, password1=password1, age=age)
        if len(password) < 8:
            return render_template('n_3.html', err="Пароль слишком короткий", email=email,
                                   name=name, surname=surname, password='', password1='', age=age)
        if password != password1:
            return render_template('n_3.html', err="Пароли не совпадают", email=email,
                                   name=name, surname=surname, password='', password1='', age=age)
        user = User.query.filter_by(
            email=email).first()
        if user:
            return render_template('n_3.html', err="Пользователь с такой почтой уже зарегистрирован")
        new_user = User(email=email, name=name, surname=surname, age=age, password=generate_password_hash(password))
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('send', email=email, reset=0))
    else:
        return render_template('n_3.html', err="")


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)