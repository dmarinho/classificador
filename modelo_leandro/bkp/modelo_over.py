import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sqlalchemy import create_engine
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import itertools
from imblearn.under_sampling import RandomUnderSampler
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

X = df['textfull']
y = df['classificado']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=np.random.randint(1, 100), shuffle=True)

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_df=0.75, min_df=3)
svc = SVC(class_weight='balanced', kernel='linear', C=1)

# Aplicar o TF-IDF nos dados de treino e teste
tf_train = tfidf.fit_transform(X_train)
tf_test = tfidf.transform(X_test)

# Aplicar o RandomUnderSampler apenas nos dados de treino
rus = RandomUnderSampler(random_state=42, sampling_strategy='majority')
X_resampled, y_resampled = rus.fit_resample(tf_train, y_train)

# Ajustar o modelo SVC com os dados reamostrados
svc.fit(X_resampled, y_resampled)

# Prever os rótulos do conjunto de teste
pred = svc.predict(tf_test)

matrix = confusion_matrix(y_true=y_test, y_pred=pred)

plt.figure(figsize=(10, 8))
plot_confusion_matrix(matrix, classes=['Concluso', 'Ausência da Parte Adversa','Redesignação','Leitura de Sentença','Acordo Realizado','Audiência Cancelada','PA Gerada Indevidamente','Desistência Ação','Sentença','Suspenso','Recurso Parte Contrária'], title='Confusion matrix')
plt.show()

pipeline = Pipeline([
    ('feature', tfidf),
    ('classifier', svc)
])

pred = svc.predict(tf_test)
model = pipeline.fit(X_train, y_train)
model.score(X_train, y_train)

dump(model, 'Over_AtasJuridicas.joblib')