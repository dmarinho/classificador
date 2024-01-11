import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold 
import joblib
import numpy as np
from sqlalchemy import create_engine


# Criar a conexão usando SQLAlchemy
engine = create_engine('mysql+mysqlconnector://root:root@localhost/classificador')

# Executar a query e carregar os dados para um DataFrame
query = "SELECT textfull, classificado FROM classificador.legado WHERE nomedoc is not null "
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

concluso_subset = df[df['classificado'] == 1].sample(n=4500, random_state=100)
other_records = df[df['classificado'] != 1]
df = pd.concat([concluso_subset, other_records])

# Divisão dos dados em conjunto de treino e teste
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df['textfull'])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['classificado'])
unique_classes = np.unique(y)

# Validação cruzada para avaliar o desempenho do modelo
#skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
scores = cross_val_score(RandomForestClassifier(n_estimators=500, max_depth=20, random_state=100), X, y, cv=5, scoring='accuracy')
print(f'Média de acurácia por validação cruzada: {scores.mean():.3f}')

# Conjunto de dados de validação
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=100)

# Treinamento do modelo
rf = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=100)
rf.fit(X_train, y_train)

# Previsões no conjunto de teste
predictions = rf.predict(X_test)

# Matriz de confusão
conf_matrix = confusion_matrix(y_test, predictions)
print("Matriz de Confusão:")
print(conf_matrix)

# Salvar o modelo
model_filename = 'modelo_randomforest.pkl'
joblib.dump(rf, model_filename)

# Salvar o codificador de rótulos
label_encoder_filename = 'label_encoder.pkl'
joblib.dump(label_encoder, label_encoder_filename)

# Salvar o vetorizador TF-IDF
tfidf_vectorizer_filename = 'tfidf_vectorizer.pkl'
joblib.dump(tfidf_vectorizer, tfidf_vectorizer_filename)
