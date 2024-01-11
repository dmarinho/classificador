import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, confusion_matrix
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import random

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

concluso_subset = df[df['classificado'] == 1].sample(n=2037, random_state=100)
other_records = df[df['classificado'] != 1]
df = pd.concat([concluso_subset, other_records])

counts = df['classificado'].value_counts().reset_index()
counts.columns = ['classificado', 'Contagem']

counts_sorted = counts.sort_values(by='Contagem', ascending=True)

df_ordenado = pd.DataFrame()

for classificado in counts_sorted['classificado']:
    df_ordenado = pd.concat([df_ordenado, df[df['classificado'] == classificado]])

df_ordenado.reset_index(drop=True, inplace=True)

# Divisão dos dados em conjunto de treino e teste
# Leandro: ngram_range=(1, 2), max_df=0.75, min_df=3
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df['textfull'])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['classificado'])
unique_classes = np.unique(y)
#
# Original: 
# Leandro: test_size=.30, random_state=random.randint(1,100), shuffle=True
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=100 )

# Calcular pesos das classes para lidar com o desbalanceamento
class_weights = class_weight.compute_class_weight('balanced', classes=unique_classes, y=y_train)

# Criar o modelo Random Forest n_estimators=200, max_depth=20, random_state=42
#RandomForestClassifier(n_estimators=100, class_weight={i: class_weights[i] for i in range(len(class_weights))})
rf = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=100, class_weight={i: class_weights[i] for i in range(len(class_weights))})

# Treinar o modelo
rf.fit(X_train, y_train)

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
