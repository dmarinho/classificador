import spacy
import mysql.connector as msql

# Carregar modelo do SpaCy para Português
nlp = spacy.load("pt_core_news_sm")

# Conectar ao banco de dados MySQL
conn = msql.connect(host='localhost',
                    database='classificador',
                    user='root',
                    password='root')
cursor = conn.cursor()

# Função para processar uma pergunta e obter a resposta
def responder_pergunta(texto, pergunta):
    doc = nlp(texto)  # Processar o texto com o SpaCy

    # Aqui, você pode implementar lógica para encontrar informações específicas, como datas, nomes de pessoas, etc.
    # Por exemplo:
    # - Para encontrar datas: Identificar tokens relevantes com atributos de data
    # - Para nomes de pessoas: Procurar por entidades do tipo "PER" (pessoas) no texto
    
    # Retornar uma resposta genérica para fins de exemplo
    return "Esta é uma resposta de exemplo baseada na pergunta: " + pergunta

# Função para buscar um texto na base de dados
def buscar_texto_por_id(id_documento):
    query = "SELECT texto FROM tabela_textos WHERE id = %s"
    cursor.execute(query, (id_documento,))
    result = cursor.fetchone()
    return result[0] if result else None

if __name__ == '__main__':
    # Exemplo: ID do documento a ser utilizado
    id_documento = 1

    # Buscar texto na base de dados
    texto_documento = buscar_texto_por_id(id_documento)

    if texto_documento:
        # Pergunta de exemplo
        pergunta = "Qual é o assunto deste documento?"

        # Obter a resposta para a pergunta
        resposta = responder_pergunta(texto_documento, pergunta)
        print("Resposta:", resposta)
    else:
        print("Documento não encontrado.")
