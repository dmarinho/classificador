
from PyPDF2 import PdfReader
import mysql.connector as msql
import spacy

 
# Carrega o modelo pré-treinado do SpaCy
nlp = spacy.load("pt_core_news_sm")

# Função para extrair texto de um PDF usando PyPDF2
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

# Função para processar a pergunta e encontrar a resposta no texto do PDF
def answer_question(question):
    # Extrai texto do PDF

    conn = msql.connect(host='localhost',
                    database='classificador',
                    user='root',
                    password='root')
    
    cursor = conn.cursor()

    query = "SELECT textfull FROM classificador.legado  Where iddocumento='8658717'  "
    cursor.execute(query)
    result = cursor.fetchall()

    text = result[0][0]
    
    # Processa o texto com SpaCy
    doc = nlp(text)
    
    # Processa a pergunta com SpaCy
    question_doc = nlp(question)
    
    # Inicializa a similaridade máxima como -1 (sem similaridade)
    max_similarity = -1
    best_answer = ""
    
    # Itera pelas sentenças no texto e encontra a mais similar à pergunta
    for sentence in doc.sents:
        similarity = sentence.similarity(question_doc)
        if similarity > max_similarity:
            max_similarity = similarity
            best_answer = sentence.text
            
    return best_answer

# Caminho para o PDF
pdf_path = 'caminho/para/seu/arquivo.pdf'

# Pergunta de exemplo
question = "Qual é o tema principal do texto?"

# Encontrar a resposta
answer = answer_question(question)
print("Resposta:", answer)
