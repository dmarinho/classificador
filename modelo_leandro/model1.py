import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sqlalchemy import create_engine
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import random
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import itertools
from sklearn.pipeline import Pipeline
from joblib import dump


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Criar a conexão usando SQLAlchemy
engine = create_engine('mysql+mysqlconnector://root:root@localhost/classificador')

# Executar a query e carregar os dados para um DataFrame
query = "SELECT textfull, classificado FROM classificador.legado WHERE nomedoc <> 'BASE2'  "
df = pd.read_sql(query, engine)

# Mapeamento das classes
mapeamento = {
    'Concluso': 1,
    'Ausência da Parte Adversa': 2,
    'Redesignação': 3,
    'Leitura de Sentença': 4,
    'Acordo Realizado': 5,
    'Audiência Cancelada': 6,
    'PA Gerada Indevidamente': 7,
    'Desistência Ação': 8,
    'Sentença': 9,
    'Suspenso': 10,
    'Recurso Parte Contrária': 11
}

# Troca dos valores na coluna 'classificado' usando o mapeamento
df['classificado'] = df['classificado'].replace(mapeamento)

# Limitar a quantidade de registros para 'Concluso' (1) a 2500

concluso_subset = df[df['classificado'] == 1].sample(n=5000, random_state=100)
other_records = df[df['classificado'] != 1]
df = pd.concat([concluso_subset, other_records])

X=df['textfull']
y=df['classificado']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=random.randint(1,100), shuffle=True )
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_df=0.75, min_df=3)
tf_train = tfidf.fit_transform(X_train)
tf_test = tfidf.transform(X_test)
svc = SVC(class_weight='balanced', kernel='linear', C=1)

svc.fit(tf_train, y_train)
pred = svc.predict(tf_test)

print("SVC-->>",classification_report(y_pred=pred, y_true=y_test))
matrix = confusion_matrix(y_pred=pred, y_true=y_test)
plt.figure(figsize=(10, 8))
plot_confusion_matrix(matrix, ['Concluso', 'Ausência da Parte Adversa','Redesignação','Leitura de Sentença','Acordo Realizado','Audiência Cancelada','PA Gerada Indevidamente','Desistência Ação','Sentença','Suspenso','Recurso Parte Contrária'])

pipeline = Pipeline([
    ('feature', tfidf),
    ('classifier', svc)
])

pred = svc.predict(tf_test)

model = pipeline.fit(X_train, y_train)

model.score(X_train, y_train)

dump(model, 'AtasJuridicas.joblib')