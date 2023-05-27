import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pymorphy2
import re
import time
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

data = pd.read_csv("data.csv")

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


def get_metrics(y_test, y_predicted):
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1


def gb_train_model(x, y):
    param_grid = {
        "learning_rate": [0.1, 0.3],
        'max_depth': range(1, 7),
        "criterion": ["friedman_mse"],
        "n_estimators": [100]
    }
    grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, scoring='accuracy',
                               cv=StratifiedKFold(n_splits=5))
    grid_search.fit(x, y)

    best_params = grid_search.best_params_
    print("Best params =", best_params)
    print("Best score =", grid_search.best_score_)
    print()
    model = grid_search.best_estimator_

    return model


data['clean'] = data['text'].map(lambda x: preprocess_text(x))
data['label_num'] = data['label'].apply(lambda x: 1 if x == 'result' else 0)
data = data[['clean', 'label_num', 'label']]
data = data.dropna()
data = data.reset_index(drop=True)

corpus = data['clean'].tolist()
labels = data['label_num'].tolist()
X_train, X_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.1, random_state=40) # 1:9

tfidf_vectorizer = TfidfVectorizer()
X_train_tf = tfidf_vectorizer.fit_transform(X_train)
X_test_tf = tfidf_vectorizer.transform(X_test)

start_time = time.time()

gb_clf_tf = gb_train_model(X_train_tf, y_train)

end_time = time.time()
total_time = end_time - start_time
print("Общее время обучения модели:", total_time, "секунд")
print()
y_predicted_tf = gb_clf_tf.predict(X_test_tf)
print("Train accuracy = ", accuracy_score(y_train, gb_clf_tf.predict(X_train_tf)))
print("Test accuracy = ", accuracy_score(y_test, y_predicted_tf))
print()
accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_tf)
print("accuracy = %.5f, precision = %.5f, recall = %.5f, f1 = %.5f" % (accuracy, precision, recall, f1))

with open('gb_clf_tf.pkl', 'wb') as f:
    pickle.dump(gb_clf_tf, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
