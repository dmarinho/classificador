import joblib
import mysql.connector as msql
import joblib
import re

def carregar_modelo(caminho_modelo):
    try:
        modelo = joblib.load(caminho_modelo)
        return modelo
    except Exception as e:
        print("Erro ao carregar o modelo:", str(e))
        return None

def classificar_texto(texto, modelo):
    try:
        resultado = modelo.predict([texto])
        return resultado
    except Exception as e:
        print("Erro ao classificar o texto:", str(e))
        return None
    
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



def classifica_nova_ata(nIdLegado):
    
    conn = msql.connect(host='localhost',
                        database='classificador',
                        user='root',
                        password='root')
        
    cursor = conn.cursor()

    query = "SELECT textfull FROM classificador.legado  Where idLegado="+str(nIdLegado)+"  "
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

    caminho_modelo = 'AtasJuridicas.joblib'
    modelo = carregar_modelo(caminho_modelo)

    for n in result:
        dDtNovaAud=''
        cHrNovaAud=''
        #nIdLegado = n[0]
        cTextFull = n[0]

        texto_para_classificar = cTextFull

        classe_predita = classificar_texto(texto_para_classificar,modelo)

        valor_correspondente = valores.get(classe_predita[0])

        if classe_predita in (3,6):
            datahora = get_date(cTextFull)

            if datahora!="":
                dDtNovaAud = datahora[0]
                cHrNovaAud = datahora[1]
        
        update_query = "update classificador.legado set resultclassificado=%s, novadtaudiencia=%s, novahraudiencia=%s  where idLegado = %s "
        cursor.execute(
        update_query, (valor_correspondente, dDtNovaAud, cHrNovaAud, str(nIdLegado)))
        conn.commit()


        print(f'O texto foi classificado como: {classe_predita}')

    cursor.close()
    conn.close()
    
    return(valor_correspondente)


