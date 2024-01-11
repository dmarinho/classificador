from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


engine = create_engine('mysql+mysqlconnector://root:root@localhost/classificador')
# Executar a query e carregar os dados para um DataFrame

query = "SELECT textfull, classificado FROM classificador.legado WHERE nomedoc is not null "
df = pd.read_sql(query, engine)

one_hot = pd.get_dummies(df["classificado"], dtype=int)
df = df.drop("classificado",axis=1)
df =df.join(one_hot)

df


cat_encoder = OneHotEncoder()

housing_cat_lhot = cat_encoder.fit_transform(df[["classificado"]])
housing_cat_lhot
housing_cat_lhot.toarray()