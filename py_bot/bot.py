import logging
import os
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Command
from aiogram.types import KeyboardButton, ReplyKeyboardMarkup
from model.predicting import predict_methaprog
from audio_recognizer.convert_transcribe import convert_to_wav, transcribe_audio


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
        "Отправьте мне текстовое или голосовое сообщение на русском языке, и я дам вердикт.\n\n"
        "Чтобы отправить голосовое сообщение, нажмите на микрофон в поле ввода сообщения.\n\n"
        "Чтобы отправить текстовое сообщение, нажмите на кнопку.\n\n"
        "Ожидаю ваши сообщения!",
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
    if transcription == "Не удалось распознать речь":
        await message.reply(f"{transcription}!")
        os.remove(file_name)
        os.remove(wav_file_name)
        return

    await message.reply(f"Ответ:  {predict_methaprog(transcription)} \n\n\nТекстовая версия:\n{transcription}")

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
