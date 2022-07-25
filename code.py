from typing import Iterable
import pandas as pd
import re
from pprint import pprint
from pymorphy2 import MorphAnalyzer
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import TruncatedSVD
import numpy as np
import optuna
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC    
from sklearn.linear_model import LogisticRegression
from statistics import median


ALLOWED_CHARS = "абвгдеёжзийклмонпрстуфхцчшщъыьэюя "

# Функция чистит строку от всех посторонних символов. Разрешенные символы передаются через allowed_chars
def filter_forbidden_chars(text : str, allowed_chars : Iterable):
    filter_fn = lambda char: char if char in allowed_chars else " "
    return re.sub(" +", " ", "".join([filter_fn(char) for char in text])).strip()

# Функция ставит все слова в строке в начальную форму
def all_words_to_infinitive(text : str, analyzer : MorphAnalyzer):
    words = text.split(" ")
    return " ".join([analyzer.parse(word)[0].normal_form for word in words]).strip()

# Функция полностью обрабатывает текст
def preprocess(data):
    data['text'] = data['text'].apply(str.lower)

    data['text'] = data['text'].apply(lambda text: filter_forbidden_chars(text, ALLOWED_CHARS))

    morpher = MorphAnalyzer()

    data['text'] = data['text'].apply(lambda text: all_words_to_infinitive(text, morpher))
    
    return data

# Функция для кросс-валидации
# Из-за очень значительного дисабаланса классов повторять приходится много раз
# Иначе результаты ненадежны и выводы о хорошем обобщении моделью данных
# сделать нельзя.
def eval_model(model, data, y):
    return median(cross_val_score(model, data, y, scoring=make_scorer(balanced_accuracy_score),
    cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=25)))

# функция для оптимизации параметров svc модели
def svc_objective(trial : optuna.Trial, data):

    model = make_pipeline(TfidfVectorizer(min_df=trial.suggest_int("min_df1", 1, 12), max_df=trial.suggest_float("max_df1", 0.5, 1.0),
                        ngram_range=(1, trial.suggest_int("max_ngram1", 1, 8)), analyzer='word'), 
                        LinearSVC(C=trial.suggest_float("C1", 0.001, 100.0), max_iter=1000,
                                multi_class=trial.suggest_categorical("multiclass1", ['crammer_singer', 'ovr']),
                                class_weight='balanced', dual=trial.suggest_categorical("dual1", [True, False])))
    
    return eval_model(model, data['text'], data['y'])

# функция для оптимизации параметров LogisticRegression
def logistic_objective(trial : optuna.Trial, data):

    model = make_pipeline(TfidfVectorizer(min_df=trial.suggest_int("min_df1", 1, 12), max_df=trial.suggest_float("max_df1", 0.5, 1.0),
                        ngram_range=(1, trial.suggest_int("max_ngram1", 1, 8)), analyzer='word'), 
                        LogisticRegression(C=trial.suggest_float("C", 0.0001, 100.0), 
                        solver=trial.suggest_categorical("solver", ['newton-cg', 'sag', 'saga', 'lbfgs']),
                        max_iter=10000,
                        n_jobs=-1, class_weight='balanced'))
    
    return eval_model(model, data['text'], data['y'])

# функция для оптимизации пары TruncatedSVD + Ridge
def lin_trunc_obj(trial : optuna.Trial, data):
    solver = trial.suggest_categorical("solver", ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'])
    positive = solver == 'lbfgs'
    model = make_pipeline(TfidfVectorizer(min_df=trial.suggest_int("min_df1", 1, 12), max_df=trial.suggest_float("max_df1", 0.5, 1.0),
                        ngram_range=(1, trial.suggest_int("max_ngram1", 1, 8)), analyzer='word'),
                        TruncatedSVD(n_components=trial.suggest_int("n_components", 2, 256), 
                        algorithm=trial.suggest_categorical("algorithm", ['arpack', 'randomized']),
                        n_iter=trial.suggest_int("n_iter", 2, 12)), 
                        StandardScaler(),
                        RidgeClassifier(alpha=trial.suggest_float("alpha", 0.001, 100.0), class_weight='balanced',
                        solver=solver, positive=positive))
    try:
        return eval_model(model, data['text'], data['y'])
    except ValueError:
        return 0.0

# Функция для подбора параметров классификатора на оснвое голосования
# участвуют все три выбранных типа моделей
def best_voting_objective(trial : optuna.Trial, data):
    make_idf = lambda index, trial: TfidfVectorizer(min_df=trial.suggest_int(f"min_df_{index}", 1, 12), max_df=trial.suggest_float(f"max_df_{index}", 0.5, 1.0),
                        ngram_range=(1, trial.suggest_int(f"max_ngram_{index}", 1, 8)), analyzer='word')

    m1 = make_pipeline(make_idf(1, trial), LinearSVC(C=trial.suggest_float("C1", 0.001, 100.0), max_iter=1000,
                                multi_class=trial.suggest_categorical("multiclass1", ['crammer_singer', 'ovr']),
                                class_weight='balanced', dual=trial.suggest_categorical("dual1", [True, False])))

    m2 = make_pipeline(make_idf(2, trial), LogisticRegression(C=trial.suggest_float("C2", 0.0001, 100.0), 
                        solver=trial.suggest_categorical("solver", ['newton-cg', 'sag', 'saga', 'lbfgs']),
                        max_iter=10000,
                        n_jobs=-1, class_weight='balanced'))

    m3 = make_pipeline(make_idf(3, trial),
                        TruncatedSVD(n_components=trial.suggest_int("n_components", 2, 256), 
                        algorithm=trial.suggest_categorical("algorithm", ['arpack', 'randomized']),
                        n_iter=trial.suggest_int("n_iter", 2, 12)), 
                        RidgeClassifier(alpha=trial.suggest_float("alpha", 0.001, 100.0), class_weight='balanced',
                        solver=trial.suggest_categorical("ridge_solver", ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']), 
                        positive=trial.params['ridge_solver'] == 'lbfgs'))

    
    model = VotingClassifier([("M1", m1), ("M2", m2), ("M3", m3)], n_jobs=3)
    
    try:
        return eval_model(model, data['text'], data['y'])
    except ValueError:
        return 0.0


if __name__ == "__main__":
    # СРАЗУ ЖЕ СБРОСИТЬ ЗАПРЕЩЕННЫЕ ДАННЫЕ И ИЗ ТРЕНИРОВКИ, И ИЗ ТЕСТА
    data = pd.read_csv("train_dataset_train.csv")[['Текст Сообщения', 'Категория']]
    data.rename(columns={"Текст Сообщения": "text", "Категория": "y"}, inplace=True)

    # Удалить категорию 12 - из-за нее не получается
    # тестировать модель через кросс-валидацию
    data = data.drop(data[data['y']==12].index)
    data.reset_index(inplace=True, drop=True)
    data = preprocess(data)

    test = pd.read_csv("test_dataset_test.csv")[['Текст Сообщения', 'id']]
    test.rename(columns={"Текст Сообщения": "text"}, inplace=True)

    test = preprocess(test)

    m1 = make_pipeline(TfidfVectorizer(min_df=6, max_df=0.6695288186828025,
                            ngram_range=(1, 4), analyzer='word'), 
                       LinearSVC(C=64.85593893692645, max_iter=1000,
                                multi_class='crammer_singer',
                                class_weight='balanced', dual=True))

    m2 = make_pipeline(TfidfVectorizer(min_df=3, max_df=0.6142292977031804,
                            ngram_range=(1, 6), analyzer='word'), 
                        LogisticRegression(C=0.1467795053973877, 
                        solver='sag',
                        max_iter=10000,
                        n_jobs=-1, class_weight='balanced'))

    m3 = make_pipeline(TfidfVectorizer(min_df=10, max_df=0.6522187041564516,
                            ngram_range=(1, 4), analyzer='word'),
                        TruncatedSVD(n_components=159, 
                        algorithm='arpack',
                        n_iter=4), 
                        RidgeClassifier(alpha=87.83655027139227, class_weight='balanced',
                        solver='saga', max_iter=10000))

    model = VotingClassifier([("M1", m1), ("M2", m2), ("M3", m3)], n_jobs=3)
    #print(eval_model(model, data['text'], data['y']))
    
    model.fit(data['text'], data['y'])
    preds = model.predict(test['text'])
    test['Категория'] = preds
    test[['id', 'Категория']].to_csv("ens_sub.csv", index=False)
    
