import logging
import os
import pickle
import re
import pymorphy2
import speech_recognition as sr
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from pydub import AudioSegment
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Command
from aiogram.types import KeyboardButton, ReplyKeyboardMarkup

import nltk

# import ssl
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
#
nltk.download('punkt')
nltk.download('stopwords')


with open('gb_clf_tf.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
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


def convert_to_wav(input_file, output_file):
    try:
        audio = AudioSegment.from_file(input_file)
        audio.export(output_file, format='wav')
        return True
    except Exception as e:
        print(f"Error occurred during audio conversion: {str(e)}")

    return False


def transcribe_audio(file_name):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_name) as source:
        audio = recognizer.record(source)
    try:
        transcription = recognizer.recognize_google(audio, language="ru-RU")
        return transcription
    except sr.UnknownValueError:
        return "Не удалось распознать речь"
    except sr.RequestError:
        return "Произошла ошибка при обращении к сервису распознавания речи"


logging.basicConfig(level=logging.INFO)
input_text = False

bot_token = ''
bot = Bot(token=bot_token)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

menu_markup = ReplyKeyboardMarkup([[KeyboardButton(text='📝 Ввести текст')]],
                                  one_time_keyboard=True,
                                  resize_keyboard=False,
                                  )


@dp.message_handler(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Привет! Я бот, который умеет определять, ориентирован ли человек на процесс или результат по тому, "
        "что и как он говорит. \n"
        "Отправьте мне текстовое или голосовое сообщение, и я дам вердикт.\n\n"
        "Чтобы отправить голосовое сообщение, нажмите на микрофон в поле ввода сообщения.\n\n"
        "Чтобы отправить текстовое сообщение, нажмите на кнопку.\n\n"
        "Ожидаю ваши голосовые и сообщения!",
        reply_markup=menu_markup
    )


@dp.message_handler(content_types=types.ContentType.VOICE)
async def handle_voice_message(message: types.Message, state: FSMContext):
    voice = message.voice
    file_id = voice.file_id
    file_unique_id = voice.file_unique_id
    file = await bot.get_file(file_id)
    file_path = file.file_path

    file_name = f"{file_unique_id}.oga"
    await bot.download_file(file_path, file_name)
    wav_file_name = f"{file_unique_id}.wav"

    if not convert_to_wav(file_name, wav_file_name):
        await message.answer(f"Что-то пошло не так. Повторите еще раз, пожалуйста")
        os.remove(file_name)
        os.remove(wav_file_name)
        return

    await message.answer(f"Аудио получено, обработка текста...\n"
                         f"Это может занять время!")

    transcription = transcribe_audio(wav_file_name)

    await message.reply(f"Ответ:  {predict_methaprog(transcription)} \n\n  Текстовая версия:\n\n{transcription}")

    os.remove(file_name)
    os.remove(wav_file_name)


@dp.message_handler()
async def echo_message(message: types.Message):
    global input_text
    if message.text == '📝 Ввести текст':
        await message.answer(f"Введенный вами далее текст будет обработан\n"
                             f"Это может занять время!")
        input_text = True
    elif input_text:
        await message.reply(f"Ответ:  {predict_methaprog(message.text)}")
        input_text = False


if __name__ == '__main__':
    from aiogram import executor

    executor.start_polling(dp, skip_updates=True)
