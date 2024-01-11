import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import joblib

# Criar a conexão usando SQLAlchemy
engine = create_engine('mysql+mysqlconnector://root:root@localhost/classificador')

# Executar a query e carregar os dados para um DataFrame
query = "SELECT textfull, classificado FROM classificador.legado WHERE statusimport = 'OK'"
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

# Limitar a quantidade de registros para 'Concluso' (1) a 7000
# modifiquei n de 5 para 7 e state de 42 para 100
concluso_subset = df[df['classificado'] == 1].sample(n=7000, random_state=100)
other_records = df[df['classificado'] != 1]
df = pd.concat([concluso_subset, other_records])

# Divisão dos dados em conjunto de treino e teste
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df['textfull'])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['classificado'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=100)

# Usar SMOTE para balancear as classes
smote = SMOTE(random_state=100)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Criar o modelo Random Forest com ajuste de hiperparâmetros
rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=100)
rf.fit(X_resampled, y_resampled)

model_filename = 'modelo_randomforest.pkl'
joblib.dump(rf, model_filename)

label_encoder_filename = 'label_encoder.pkl'
joblib.dump(label_encoder, label_encoder_filename)

tfidf_vectorizer_filename = 'tfidf_vectorizer.pkl'
joblib.dump(tfidf_vectorizer, tfidf_vectorizer_filename)

# Fazer previsões no conjunto de teste
predictions = rf.predict(X_test)

# Calcular a acurácia do modelo
accuracy = accuracy_score(y_test, predictions)
print(f'Acurácia do modelo: {accuracy}')

# Calcular e imprimir a matriz de confusão
conf_matrix = confusion_matrix(y_test, predictions)
print("Matriz de Confusão:")
print(conf_matrix)
