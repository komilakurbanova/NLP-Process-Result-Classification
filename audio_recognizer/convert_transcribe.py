import speech_recognition as sr
from pydub import AudioSegment


def convert_to_wav(input_file, output_file):
    try:
        audio = AudioSegment.from_file(input_file)
        audio.export(output_file, format='wav')
        return True
    except Exception as e:
        print(f"Error occurred during audio conversion: {str(e)}")

    return False


def convert_to_raw_data(audio_file):
    try:
        audio = AudioSegment.from_file(audio_file)
        audio = audio.set_frame_rate(16000)
        audio = audio.set_channels(1)
        raw_data = audio.raw_data
        return raw_data
    except Exception as e:
        return None


def transcribe_audio_raw(audio_data):
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


def transcribe_audio(file_name):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_name) as source:
        audio = recognizer.record(source)
    try:
        transcription = recognizer.recognize_google(audio, language="ru-RU")
        return transcription
    except Exception as e:
        return "Не удалось распознать речь"
