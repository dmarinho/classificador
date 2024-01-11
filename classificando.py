import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import mysql.connector as msql
import re


# Carregar o modelo, LabelEncoder e TF-IDFVectorizer
model_filename = 'modelo_randomforest.pkl'
label_encoder_filename = 'label_encoder.pkl'
tfidf_vectorizer_filename = 'tfidf_vectorizer.pkl'

rf_model = joblib.load(model_filename)
label_encoder = joblib.load(label_encoder_filename)
tfidf_vectorizer = joblib.load(tfidf_vectorizer_filename)

def classificar_texto(novo_texto):
    # Transformar o novo texto usando o vetorizador TF-IDF
    novo_texto_transformado = tfidf_vectorizer.transform([novo_texto])
    # Fazer a previsão usando o modelo carregado
    resultado = rf_model.predict(novo_texto_transformado)
    # Mapear o resultado de volta para a classe original usando o LabelEncoder inverso
    classe_mapeada = label_encoder.inverse_transform(resultado)

    return classe_mapeada[0]

def converter_hora(texto):
    # Verificar se o texto contém "99 99h" e substituir por "99:99"
    texto = re.sub(r'(\d{2})\s(\d{2})h', r'\1:\2', texto)

    # Verificar se o texto contém "99h99min" e substituir por "99:99"
    texto = re.sub(r'(\d{2})h(\d{2})min', r'\1:\2', texto)

    # Verificar se o texto contém "99h" e substituir por "99:00"
    texto = re.sub(r'(\d{2})h', r'\1:00', texto)

    return texto


def get_date(textfull):
    data_formatada = ""
    hora_formatada = ""

    meses = "(janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)"

    # Padrão para identificar datas no formato xx/xx/xxxx ou xx.xx.xxxx
    padrao_data = r'\b(?:remarca|remarco|remarcada|nova\sdata\se\sHORA\sde\saudiência|novo\sdia\sde\saudiência)\s*[^.?!]*?\b(\d{2}[/. ]\d{2}[/. ]\d{4})\b'

    padrao_hora = r'(?:(?<=\b\d{2}[/. ]\d{2}[/. ]\d{4}\b)|(?<=\bas\s))\s*((?:\d{2}:\d{2}(?:hs?| horas?)?)|(?:nove horas)|(?:\d{1,2}h\d{2}min)|(?:\d{2}h\d{2}min)|(?:(?:\d{1,2}|\d{2})\s?\d{2}h))\b'

    # Buscar datas e horas
    datas_encontradas = re.findall(padrao_data, textfull, re.IGNORECASE)
    horas_encontradas = re.findall(padrao_hora, textfull, re.IGNORECASE)

    if len(datas_encontradas) != 0:
        data_formatada = re.sub(r'\s', '/', datas_encontradas[0])
        hora_formatada = converter_hora(horas_encontradas[0])

    return (data_formatada,hora_formatada)



conn = msql.connect(host='localhost',
                    database='classificador',
                    user='root',
                    password='root')
    
cursor = conn.cursor()

query = "SELECT iddocumento, textfull FROM classificador.legado  Where nomedoc='BASE2'  "
cursor.execute(query)
result = cursor.fetchall()

valores = {
    1: 'Concluso',
    2: 'Ausência da Parte Adversa',
    3: 'Redesignação',
    4: 'Leitura de Sentença',
    5: 'Acordo Realizado',
    6: 'Audiência Cancelada',
    7: 'PA Gerada Indevidamente',
    8: 'Desistência Ação',
    9: 'Sentença',
    10: 'Suspenso',
    11: 'Recurso Parte Contrária'
}

for n in result:
    dDtNovaAud=''
    cHrNovaAud=''
    n0 = n[0]
    n1 = n[1]

    texto_para_classificar = n1

    classe_predita = classificar_texto(texto_para_classificar)

    valor_correspondente = valores.get(classe_predita)

    if classe_predita in (3,6):
        datahora = get_date(n1)

        if datahora!="":
            dDtNovaAud = datahora[0]
            cHrNovaAud = datahora[1]
     
    update_query = "update classificador.legado set resultclassificado=%s, novadtaudiencia=%s, novahraudiencia=%s  where iddocumento = %s "
    cursor.execute(
    update_query, (valor_correspondente, dDtNovaAud, cHrNovaAud, str(n0)))
    conn.commit()


    print(f'O texto foi classificado como: {classe_predita}')

cursor.close()
conn.close()


"""
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
"""
