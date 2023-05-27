from flask import Flask, request, render_template, redirect, url_for, flash
import speech_recognition as sr
from pydub import AudioSegment
from model.predicting import predict_methaprog
from audio_recognizer.convert_transcribe import convert_to_raw_data, transcribe_audio_raw

app = Flask(__name__)
app.secret_key = 'your_secret_key'


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'audio_file' in request.files:
        audio_file = request.files['audio_file']

        audio_data = convert_to_raw_data(audio_file)

        if audio_data is None:
            return redirect_with_error('Ошибка конвертации в формат аудио')

        transcription = transcribe_audio_raw(audio_data)
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
