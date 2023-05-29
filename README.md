# Цифровой анализ речи: выявление нацеленности интервьюируемого на результат или процесс

Добро пожаловать в репозиторий курсового проекта, посвященного бинарной классификации русскоязычного текста (и аудиозаписей) на основе метапрограммы "ориентация на результат или процесс". В данном проекте была разработана модель, способная определять, на что больше ориентирован автор текста: на результат или на процесс. Для более полного погружения рекомендуется ознакомиться с [отчетом по курсовой работе](https://www.overleaf.com/read/thyncrpfszsb)

## Описание проекта
Цель данного проекта состоит в исследовании ориентации автора текста (или аудиодорожки) на результат или процесс с использованием анализа речи и текстовых данных на русском языке. Разработанная модель на основе лингвистического анализа определяет, как автор воспринимает себя в контексте общества.

Данная модель может быть полезна в подборе и найме сотрудников, когда требуется оценить, какой подход лучше соответствует определенной должности. Например, HR-специалисты могут задаться вопросом: "В чем кандидат лучше: в целеполагании и достижении результатов или в планомерной реализации?" Разработанная в рамках проекта модель поможет в автоматизации такого отбора.

Для удобного использования модели были разработаны два интерфейса:

1. [Telegram-бот](http://t.me/ResultProcessModelBot): Бот на aiogram, который позволяет отправлять текстовые запросы и голосовые сообщения. Голосовые сообщения автоматически преобразуются в текст с помощью службы распознавания речи Google Cloud STT, после чего модель классифицирует полученный текстовый запрос и отправляет ответ пользователю.

2. Веб-сайт: Разработан сайт на Flask, который принимает текстовые запросы и аудиофайлы в форматах `.mp3`, `.wav` и `.flac`. Аудиофайлы также преобразуются в текст с помощью Google Cloud STT, затем модель классифицирует текст и отображает ответ на сайте для пользователя.

## Структура репозитория

### Модель Gradient Boosting

- `model_development.ipynb` Файл Jupyter Notebook, содержащий код и описание всех экспериментов, проведенных при выборе и создании модели.
- `model/` Директория с файлами, связанными с моделью:
  - `data.csv` Размеченная выборка до обработки.
  - `gradient_boost.py` Код для обучения модели градиентного бустинга, включая предобработку данных и сохранение необходимых для внешнего использования модели файлов.
  - `predicting.py` Код для получения предсказаний от модели.
  - `gb_clf_tf.pkl` и `vectorizer.pkl` Выгруженная модель и векторизатор соответственно.
  
### Telegram-бот

- `py_bot/` Директория с файлами, относящимися к Telegram-боту:
  - `bot.py` Код Telegram-бота на основе библиотеки aiogram.
- `Dockerfile`: Файл Docker для создания контейнера с ботом.

### Веб-сайт

- `flask_website/` Директория с файлами, относящимися к веб-сайту:
  - `app.py` Код Flask-приложения, обрабатывающего текстовые запросы и аудиофайлы.
  - `templates/`: Директория с HTML-шаблонами для веб-страниц.

### Работа с аудио

- `audio_recognizer/` Директория с файлами, относящимися к транскрибации аудиофайлов.
  - `convert_transcribe.py` Код конвертации аудиофайлов в необходимый для транскрибации формат и непосредственно перевод в текстовый формат.


## Установка и запуск

1. Клонируйте репозиторий на свой локальный компьютер:
`https://github.com/komilakurbanova/NLP-Process-Result-Classification.git`

2. Установите необходимые зависимости:\
`pip install -r requirements.txt`\
`apt install ffmpeg` или другая команда, в зависимости от операционной системы

Запуск осуществляйте из директории NLP-Process-Result-Classification.

- Запуск обучения модели: `python -m model.gradient_boost`

- Запуск веб-сайта:  `python -m flask_website.app`

- Запуск Telegram-бота:  `python -m py_bot.bot`, предварительно добавив токен бота в окружение.

> Вы также можете использовать Docker для упрощения установки и запуска бота:
>- Клонируйте репозиторий:
> `docker pull komilakurbanova03/nlp_bot:tag`
>- Запустите контейнер с ботом:
>
> `docker run -d --name your_container_name -e BOT_TOKEN=your_bot_token komilakurbanova03/nlp_bot:tag`
> 
>  нужно заменить `your_container_name` и `your_bot_token` на ваши данные

## Примеры использования
### [Пример работы бота](https://disk.yandex.ru/i/hb2GrH7yIMKZVQ)
### [Пример работы сайта](https://disk.yandex.ru/i/5nZN6C4NfD0R6A)

Входные данные должны быть на русском языке.

- Ориентация на результат:
```
from model.predicting import predict_methaprog
text = "Я решительно стремлюсь к достижению поставленных целей и не останавливаюсь до тех пор, пока не достигну желаемого результата."
print(predict_methaprog(text))
```
> Вывод: `result`

- Ориентация на процесс:

```
from model.predicting import predict_methaprog
text = "Я наслаждаюсь процессом работы и постоянным совершенствованием, полностью погружаясь в каждый шаг"
print(predict_methaprog(text))
```

> Вывод: `process`


## Лицензия

Данный проект находится под [лицензией MIT](https://github.com/komilakurbanova/NLP-Process-Result-Classification/blob/main/LICENSE)


## Авторство 

Проект "Цифровой анализ речи: выявление нацеленности интервьюируемого на результат или процесс" полностью разработан и поддерживается [Курбановой Комилой](https://github.com/komilakurbanova) под руководством [Передерина Д.А.](https://www.hse.ru/org/persons/225271929) 

Если у вас есть вопросы или предложения по улучшению проекта, пожалуйста, свяжитесь со мной через [Telegram](https://t.me/kvrmalin) или [электронную почту](krkurbanova@edu.hse.ru).

Выражаю благодарность научному руководителю за внесенный вклад в развитие проекта, включая предоставление данных, обратную связь и поддержку.
