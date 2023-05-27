from flask import Flask, request, render_template, redirect, url_for, flash
import speech_recognition as sr
from pydub import AudioSegment
import pickle
import re
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pymorphy2
import nltk

# import ssl
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
app.secret_key = 'your_secret_key'

with open('../model/gb_clf_tf.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

classes = {0: 'process', 1: 'result'}


def preprocess_text(text):
    morph = pymorphy2.MorphAnalyzer()

    text = text.lower()

    text = re.sub(r"[^\w\s]", "", text)

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('russian'))

    lemmatized_tokens = []
    for token in tokens:
        parsed_token = morph.parse(token)[0]
        if 'VERB' in parsed_token.tag:
            lemmatized_tokens.append(parsed_token.tag.aspect)
        if parsed_token.normal_form.isdigit():
            lemmatized_tokens.append('num')

        if not parsed_token.normal_form.isdigit() and parsed_token.normal_form not in stop_words:
            lemma = SnowballStemmer('russian').stem(parsed_token.normal_form)
            lemmatized_tokens.append(lemma)

    processed_text = ' '.join(lemmatized_tokens)
    return processed_text


def predict_methaprog(sentence):
    sentence_processed = preprocess_text(sentence)
    sentence_tf = tfidf_vectorizer.transform([sentence_processed])
    prediction = model.predict(sentence_tf)
    return classes[prediction[0]]


def convert_to_wav(audio_file):
    try:
        audio = AudioSegment.from_file(audio_file)
        audio = audio.set_frame_rate(16000)
        audio = audio.set_channels(1)
        raw_data = audio.raw_data
        return raw_data
    except Exception as e:
        return None


def transcribe_audio(audio_data):
    r = sr.Recognizer()
    audio = sr.AudioData(audio_data, sample_rate=16000, sample_width=2)
    try:
        result = r.recognize_google(audio, language='ru-RU', show_all=True)
        if not isinstance(result, dict) or len(result.get("alternative", [])) == 0:
            raise sr.UnknownValueError()
        best_alternative = result["alternative"][0]["transcript"]
        return best_alternative
    except Exception as e:
        return None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'audio_file' in request.files:
        audio_file = request.files['audio_file']

        audio_data = convert_to_wav(audio_file)

        if audio_data is None:
            return redirect_with_error('Ошибка конвертации в формат аудио')

        transcription = transcribe_audio(audio_data)
        if transcription is None:
            return redirect_with_error('Ошибка распознавания речи')
        return redirect(url_for('text_page', text=transcription))

    return redirect_with_error('Не найден аудиофайл')


@app.route('/text')
def text_page():
    text = request.args.get('text')
    return render_template('text.html', text=text)


@app.route('/result')
def result_page():
    text = request.args.get('text')
    if text:
        result = predict_methaprog(text)
        return render_template('result.html', text=result)

    return redirect_with_error('Не указан текст.')


@app.route('/redirect', methods=['POST'])
def redirect_page():
    text = request.form['text_input']
    if text:
        return redirect(url_for('result_page', text=text))

    return redirect_with_error('Не указан текст.')


@app.route('/manual', methods=['POST'])
def manual():
    if 'text_input' in request.form:
        text = request.form['text_input']
        return redirect(url_for('text_page', text=text))

    return redirect_with_error('Не указан текст.')


def redirect_with_error(error_message):
    flash(error_message, 'error')
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run()
