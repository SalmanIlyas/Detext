from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
import pandas as pd
import numpy as np
import joblib
import logging
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy

# Inisialisasi aplikasi Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'  # Gantilah dengan secret key yang aman
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'  # Gunakan SQLite untuk contoh ini
db = SQLAlchemy(app)

# Atur logging untuk debug
logging.basicConfig(level=logging.DEBUG)

# Memuat model
model_path = 'model/emotion_classifier.pkl'  # Sesuaikan path model Anda
pipe_lr = joblib.load(open(model_path, "rb"))

# Dictionary untuk emosi dan emoji
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨",
    "happy": "🤗", "sad": "😔", "shame": "😳", "surprise": "😮"
}

# Pre-processing setup
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    tokens = [word for word in text.split() if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Prediction functions
def predict_emotions(docx):
    preprocessed_text = preprocess_text(docx)
    logging.debug(f'Preprocessed text: {preprocessed_text}')
    results = pipe_lr.predict([preprocessed_text])
    return results[0]

def get_prediction_proba(docx):
    preprocessed_text = preprocess_text(docx)
    results = pipe_lr.predict_proba([preprocessed_text])
    return results

# Model User untuk database
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
           
            return redirect(url_for('index'))
        else:
            flash('Login failed. Check your username or password.', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        
        # Periksa apakah email sudah ada
        if User.query.filter_by(email=email).first():
            flash('Email is already registered. Please use a different email.', 'error')
            return redirect(url_for('register'))
        
        # Periksa apakah username sudah ada
        if User.query.filter_by(username=username).first():
            flash('Username is already taken. Please choose a different username.', 'error')
            return redirect(url_for('register'))
        
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
      
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/app_page', methods=['GET', 'POST'])
def app_page():
    if request.method == 'POST':
        raw_text = request.form['raw_text']

        logging.debug(f'Raw text: {raw_text}')

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        logging.debug(f'Prediction: {prediction}')
        logging.debug(f'Probability: {probability}')

        emoji = emotions_emoji_dict.get(prediction, '🤔')  # Default emoji if prediction not found
        confidence = np.max(probability) * 100
        confidence = f"{confidence:.2f}%"

        # Prepare data for Altair
        proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
        proba_df_clean = proba_df.T.reset_index()
        proba_df_clean.columns = ["emotion", "probability"]
        data = proba_df_clean.to_dict(orient='records')

        return render_template('app_page.html', text=raw_text, prediction=prediction, emoji=emoji,
                               confidence=confidence, data=data)
    return render_template('app_page.html')

@app.route('/predict', methods=['POST'])
def predict():
    raw_text = request.form['raw_text']
    prediction = predict_emotions(raw_text)
    probability = get_prediction_proba(raw_text)

    response = {
        'text': raw_text,
        'prediction': prediction,
        'emoji': emotions_emoji_dict.get(prediction, '🤔'),
        'confidence': np.max(probability),
        'probabilities': probability.tolist(),
        'classes': pipe_lr.classes_.tolist()
    }

    return jsonify(response)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Membuat database dan tabel jika belum ada
    app.run(debug=True)
