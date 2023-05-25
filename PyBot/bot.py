import logging
import os
import speech_recognition as sr
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Command
from pydub import AudioSegment
from aiogram.types import ReplyKeyboardRemove, ReplyKeyboardMarkup, KeyboardButton

logging.basicConfig(level=logging.INFO)
input_text = False

bot_token = ''
bot = Bot(token=bot_token)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

menu_markup = ReplyKeyboardMarkup([[KeyboardButton(text='📝 Ввести текст')]],
                                  one_time_keyboard=True,
                                  resize_keyboard=True,
                                  )

# Устанавливаем команду-хендлер /start
@dp.message_handler(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Привет! Я бот, который умеет транскрибировать голосовые сообщения. Просто отправь мне голосовое сообщение, и я отправлю тебе его текстовую версию.\n\n"
        "Чтобы отправить голосовое сообщение, нажми на микрофон в поле ввода сообщения.\n\n"
        "Чтобы отправить текстовое сообщение, нажми на кнопку.\n\n"
        "Ожидаю твои голосовые и сообщения!",
        reply_markup=menu_markup
    )

# Устанавливаем хендлер на голосовые сообщения
@dp.message_handler(content_types=types.ContentType.VOICE)
async def handle_voice_message(message: types.Message, state: FSMContext):
    # Получаем информацию о файле голосового сообщения
    voice = message.voice
    file_id = voice.file_id
    file_unique_id = voice.file_unique_id
    file = await bot.get_file(file_id)
    file_path = file.file_path

    # Скачиваем файл голосового сообщения в формат WAV
    file_name = f"{file_unique_id}.oga"
    await bot.download_file(file_path, file_name)
    wav_file_name = f"{file_unique_id}.wav"
    convert_to_wav(file_name, wav_file_name)

    await message.answer(f"Аудио получено, обработка текста...\n"
                         f"Это может занять время!")

    transcription = transcribe_audio(wav_file_name)

    await message.reply(f"{transcription}")

    os.remove(file_name)
    os.remove(wav_file_name)

# Устанавливаем хендлер на текстовые сообщения
@dp.message_handler()
async def echo_message(message: types.Message):
    global input_text
    if message.text == '📝 Ввести текст':
        await message.answer(f"Введенный вами далее текст будет обработан\n"
                             f"Это может занять время!")
        input_text = True
    elif input_text:
        await message.answer(f"Вы ввели:\n\n{message.text}")
        input_text = False

def convert_to_wav(input_file, output_file):
    audio = AudioSegment.from_file(input_file)
    audio.export(output_file, format='wav')

def transcribe_audio(file_name):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_name) as source:
        audio = recognizer.record(source)
    try:
        transcription = "Текстовая версия:\n\n" + recognizer.recognize_google(audio, language="ru-RU")
        return transcription
    except sr.UnknownValueError:
        return "Не удалось распознать речь"
    except sr.RequestError:
        return "Произошла ошибка при обращении к сервису распознавания речи"

if __name__ == '__main__':
    from aiogram import executor
    executor.start_polling(dp, skip_updates=True)
