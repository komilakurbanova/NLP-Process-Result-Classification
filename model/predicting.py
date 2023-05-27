import pickle

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pymorphy2
import re
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


with open('model/gb_clf_tf.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/vectorizer.pkl', 'rb') as f:
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
