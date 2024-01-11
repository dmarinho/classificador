import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


# Criar a conexão usando SQLAlchemy
engine = create_engine('mysql+mysqlconnector://root:root@localhost/classificador')

# Executar a query e carregar os dados para um DataFrame
query = "SELECT textfull, classificado FROM classificador.legado WHERE nomedoc is not null "
#"SELECT textfull, classificado FROM classificador.legado WHERE statusimport = 'OK' "
df = pd.read_sql(query, engine)

# Suponhamos que você já tenha seu DataFrame df com as colunas 'textfull' e 'classificador'
# e tenha carregado os dados necessários nele

""" Concluso
Ausência da Parte Adversa
Redesignação
Leitura de Sentença
Acordo Realizado
Audiência Cancelada
PA Gerada Indevidamente
Desistência Ação
Sentença
Suspenso
Recurso Parte Contrária """

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


# Dividindo os dados em conjunto de treino e teste
#X = df['textfull']
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df['textfull'])


#y = df['classificado']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['classificado'])
unique_classes = np.unique(y)

# Divisão estratificada para garantir a representação proporcional das classes nos conjuntos de treino e teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

# Calcular pesos das classes para lidar com o desbalanceamento
#class_weights = class_weight.compute_class_weight('balanced', classes=sorted(y.unique()), y=y_train)
class_weights = class_weight.compute_class_weight('balanced', classes=unique_classes, y=y_train)

# Criando o modelo Random Forest
#rf = RandomForestClassifier(n_estimators=100, class_weight={i: class_weights[i-1] for i in sorted(y.unique())})
 
rf = RandomForestClassifier(n_estimators=100, class_weight={i: class_weights[i] for i in range(len(class_weights))})

# Treinando o modelo
rf.fit(X_train, y_train)

model_filename = 'modelo_randomforest.pkl'  # Substitua pelo caminho desejado
joblib.dump(rf, model_filename)

label_encoder_filename = 'label_encoder.pkl'
joblib.dump(label_encoder, label_encoder_filename)

tfidf_vectorizer_filename = 'tfidf_vectorizer.pkl'
joblib.dump(tfidf_vectorizer, tfidf_vectorizer_filename)

# Fazendo previsões no conjunto de teste
predictions = rf.predict(X_test)

# Calculando a acurácia do modelo
accuracy = accuracy_score(y_test, predictions)
print(f'Acurácia do modelo: {accuracy}')
