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
        return render_template('index.html')


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
        # добавить условие для сложностей паролей
        # return render_template('new_password.html', psw1='', psw2='')
        email = session.get('email', None)
        user = User.query.filter_by(email=email).first()
        if user != '':
            user.password = psw1
            db.session.commit()
        return redirect(url_for('login'))
    else:
        return render_template('new_password.html', psw1='', psw2='')


CODE = 0


@app.route('/send', methods=['GET', 'POST'])
def send():
    if request.method == 'POST':
        global CODE
        print(request.form)
        email = request.form['mail']
        unic_code = request.form['unik_cod']
        if email == '':
            return render_template('password.html', flag=False, err="Введите почту")
        if unic_code == '':
            flag = True
            CODE = randint(1000, 9999)
            message = (f'''Здравствуйте!
            Вы получили это письмо, потому что мы получили запрос на сброс пароля для вашей учетной записи.
            Специальный код для сброса пароля: {CODE}
            Если вы не запрашивали сброс пароля, никаких дальнейших действий не требуется.

            С Уважением,
            EdTech платформа''')
            print(send_email(message=message, adress=email))
            return render_template('password.html', flag=True, err="Код отправлен", email=email)
        else:
            if int(unic_code) == CODE:
                CODE = 0
                session['email'] = email
                return redirect(url_for('new_password'))
    else:
        return render_template('password.html', flag=False, err="")


@app.route('/profile')
@login_required
def profile():
    r = ""
    if current_user.role == 0:
        r = "Студент"
    elif current_user.role == 1:
        r = "Наставник"
    elif current_user.role == 2:
        r = "Администратор"

    return render_template('profile.html', name=current_user.name, surname=current_user.surname,
                           surname1=current_user.surname1, email=current_user.email, image='img/' + current_user.image,
                           role=r)


@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    if request.method == 'POST':
        print(request.form)
        if 'name' in tuple(request.form):
            name = request.form.get('name')
            cu = User.query.filter_by(id=current_user.id).first()
            cu.name = name
            db.session.commit()
        if 'surname' in tuple(request.form):
            surname = request.form.get('surname')
            cu = User.query.filter_by(id=current_user.id).first()
            cu.surname = surname
            db.session.commit()
        if 'surname1' in tuple(request.form):
            surname1 = request.form.get('surname1')
            cu = User.query.filter_by(id=current_user.id).first()
            cu.surname1 = surname1
            db.session.commit()
        if 'upload_image' in tuple(request.form):
            file = request.files['file']
            if file.filename != '':

                if file.filename.split(".")[-1].upper() not in "PNG, JPEG, GIF, RAW, TIFF, BMP, PSD":
                    return render_template('edit_profile.html', name=current_user.name, surname=current_user.surname,
                                           surname1=current_user.surname1, email=current_user.email,
                                           image='img/' + current_user.image, error="Ошибка расширения")
                cu = User.query.filter_by(id=current_user.id).first()
                cu.image = file.filename
                db.session.commit()
                # безопасно извлекаем оригинальное имя файла
                filename = file.filename
                # сохраняем файл
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('edit_profile.html', name=current_user.name, surname=current_user.surname,
                                   surname1=current_user.surname1, email=current_user.email,
                                   image='img/' + current_user.image)
        else:
            return redirect(url_for('profile'))
    else:

        return render_template('edit_profile.html', name=current_user.name, surname=current_user.surname,
                               surname1=current_user.surname1, email=current_user.email,
                               image='img/' + current_user.image)


@app.route('/entrance', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        print(request.form)
        email = request.form.get('mail')
        password = request.form.get('password')
        remember = request.form.get('rememder')
        if remember:
            remember = True
        else:
            remember = False
        if not email or not password:
            return render_template('entrance.html', err='Заполнены не все поля')
        if '@' not in email:
            return render_template('entrance.html', err='Неправильно указана почта')
        if len(password) < 8:
            return render_template('entrance.html', err='Слабый пароль')
        user = User.query.filter_by(email=email).first()
        if not user:
            return render_template('entrance.html', err='Почта не зарегистрирована')
        user = User.query.filter((User.email == email) & (User.password == password)).first()
        if user:
            login_user(user, remember=remember)
            return redirect(url_for('index'))
        else:
            return render_template('entrance.html', err='Проверьте данные')
    else:
        return render_template('entrance.html', err='')


@app.route('/signup', methods=['POST', 'GET'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        name = request.form.get('name')
        surname = request.form.get('surname')
        password = request.form.get('password')
        password1 = request.form.get('password1')
        age = request.form.get('age')
        role = 0
        image = 'profile-rev.png'
        if not email or not name or not surname or not password or not password1 or not age:
            return render_template('n_3.html', err="Заполнены не все поля")
        if '@' not in email:
            return render_template('n_3.html', err="Некорректная почта")
        if len(password) < 8:
            return render_template('n_3.html', err="Пароль слишком короткий")
        if password != password1:
            return render_template('n_3.html', err="Пароли не совпадают")
        user = User.query.filter_by(
            email=email).first()
        if user:
            return render_template('n_3.html', err="Пользователь с такой почтой уже зарегистрирован")
        new_user = User(email=email, name=name, surname=surname, surname1=surname1, password=password, role=int(role),
                        image=image)

        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('profile'))
    else:
        return render_template('n_3.html', err="")


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
