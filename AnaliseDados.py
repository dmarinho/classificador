import pandas as pd
from sqlalchemy import create_engine
from ydata_profiling import ProfileReport

# Criar a conexão usando SQLAlchemy
engine = create_engine('mysql+mysqlconnector://root:root@localhost/classificador')

# Executar a query e carregar os dados para um DataFrame
query = "SELECT textfull, classificado FROM legado WHERE statusimport ='OK' "
df = pd.read_sql(query, engine)

profile = ProfileReport(df)
#profile.to_widgets()

profile.to_file("your_report.html")

# Verificar a forma do DataFrame
print(df.shape)