from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_required, current_user, logout_user, login_user
import os
import smtplib
from email.mime.text import MIMEText
from random import randint

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



class User(UserMixin, db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    name = db.Column(db.String(100))
    surname = db.Column(db.String(100))
    age = db.Column(db.String(100))



with app.app_context():
    db.create_all()


def send_email(message, adress):
    sender = "gulovskiu@gmail.com"
    password = "nwjcfhzloyluetwv"
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    try:
        server.login(sender, password)
        msg = MIMEText(message)
        msg["Subject"] = "Восстановление пароля"
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
        pass
    else:
        if current_user.is_authenticated:
            flag_user = True
        else:
            flag_user = False
        return render_template('index.html', flag=flag_user)


@app.route('/new_password', methods=['GET', 'POST'])
def new_password():
    if request.method == 'POST':
        psw1 = request.form['psw1']
        psw2 = request.form['psw2']
        if not psw1 or not psw2:
            return render_template('new_password.html', err='Заполните все поля', psw1=psw1, psw2=psw2)
        if len(psw1) < 8:
            return render_template('new_password.html', err='Пароль слишком маленький', psw1=psw1, psw2=psw2)
        if psw1 != psw2:
            return render_template('new_password.html', err='Пароли различаются', psw1=psw1, psw2=psw2)
        email = session.get('email', None)
        user = User.query.filter_by(email=email).first()
        if user != '':
            user.password = psw1
            db.session.commit()
        return redirect(url_for('send', email=email))
    else:
        return render_template('new_password.html', psw1='', psw2='')


CODE = 0


@app.route('/send', methods=['GET', 'POST'])
def send():
    email = request.args.get('email')
    if request.method == 'POST':
        global CODE
        email = request.form['mail']
        unic_code = request.form['unik_cod']
        if email == '':
            return render_template('password.html', flag=False, err="Введите почту")
        if unic_code == '':
            CODE = randint(1000, 9999)
            message = (f'''Здравствуйте!
            Вы получили это письмо, потому что мы получили запрос на сброс пароля для вашей учетной записи.
            Специальный код для сброса пароля: {CODE}
            Если вы не запрашивали сброс пароля, никаких дальнейших действий не требуется.

            С Уважением,
            EdTech платформа''')
            send_email(message=message, adress=email)
            return render_template('password.html', flag=True, err="Код отправлен", email=email)
        else:
            if int(unic_code) == CODE:
                CODE = 0
                session['email'] = email
                return redirect(url_for('index'))
    else:
        return render_template('password.html', flag=False, err="", email=email)


@app.route('/profile')
def profile():
    return render_template('profile.html', name=current_user.name, surname=current_user.surname,
                           age=current_user.age, email=current_user.email, image='static/image/profile_rev.png')

@app.route('/watch')
def movie():
    return render_template('movie.html')

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
        user = User.query.filter((User.email == email) & (User.password == password)).first()
        if user:
            login_user(user, remember=remember)
            return redirect(url_for('index'))
        else:
            return render_template('entrance.html', err='Проверьте данные', email=email,
                                   password=password, remember=remember)
    else:
        return render_template('entrance.html', err='')


@app.route('/recomendations', methods=['GET', 'POST'])
def recomendations():
    if request.method == 'POST':
        pass
    else:
        return render_template('recomendations.html')


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
        new_user = User(email=email, name=name, surname=surname, age=age, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('send', email=email))
    else:
        return render_template('n_3.html', err="")


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)