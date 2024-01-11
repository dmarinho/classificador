import pandas as pd
import mysql.connector as msql
import PyPDF2
import pytesseract
import os
import re
import nltk
import unidecode
import string

import chardet
from PIL import Image
from PIL import ImageEnhance
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from unicodedata import normalize
from joblib import Parallel, delayed
from bs4 import BeautifulSoup

conn = msql.connect(host='localhost',
                    database='classificador',
                    user='root',
                    password='root')
    
cursor = conn.cursor()


def localizar_arquivos_por_parte_do_nome(parte_do_nome):
    diretorio = r'C:\\_diogenes_\\LEXi\\AtasRead_legado\\melhoria_textos\\api\\tmp\\'
    caminho_completo = ''
    for pasta_raiz, subpastas, arquivos in os.walk(diretorio):
        for arquivo in arquivos:
            if parte_do_nome in arquivo:
                caminho_completo = os.path.join(pasta_raiz, arquivo)
   
    return (caminho_completo)


def pdf_to_jpg(pdf_file, output_folder):

    # try:
    
    pdf = PdfReader(open(pdf_file, 'rb'))
    num_pages = len(pdf.pages)
    

    for page_num in range(num_pages):
        images = convert_from_path(
            pdf_file,
            first_page=page_num + 1,
            last_page=page_num + 1,
            poppler_path=r'C:\\_diogenes_\\ClassificadorAtas\\poppler-23.11.0\\Library\\bin'
        )

        # 'C:\\ProgramData\\anaconda3\\Library\\bin'

        for i, image in enumerate(images):
            image.save(os.path.join(output_folder,
                       f'page_{page_num + 1}_{i + 1}.jpg'), 'JPEG')
    # except:
    #     print("ERRO: Arquivo no formato diferete ou corrompto." )
    #     print(pdf_file)


def extrair_texto_do_pdf(pdf_path):
    texto = ""
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            texto += page.extract_text()
    print("extraio por texto")
    return texto


def remover_acentos(txt):
    try:
        r = normalize('NFKD', txt).encode('ASCII', 'ignore').decode('ASCII')
    except:
        print("erro, econde")
    return r 


def remove_punctuation(text):
    if type(text) == float:
        return text
    ans = ""
    for i in text:
        if i not in string.punctuation:
            ans += i
    return ans


def limpar_texto(texto):
    # Colocar todas as palavras em minúsculas
    texto = texto.lower()
    # texto = remover_acentos(texto)
    # texto = remove_punctuation(texto)

    # Remover caracteres especiais e pontuações, incluindo acentos
    texto = unidecode.unidecode(texto)
    # Modificando a expressão regular para manter números
    texto = re.sub(r'[^\w\s]', ' ', texto)

    # Tokenizar o texto em palavras
    palavras = word_tokenize(texto)

    # Remover stopwords em português
    stopwords_portugues = set(stopwords.words('portuguese'))
    palavras_sem_stopwords = [
        palavra for palavra in palavras if palavra not in stopwords_portugues]

    # Juntar as palavras novamente em um texto limpo
    texto_limpo = ' '.join(palavras_sem_stopwords)

    return texto_limpo

def processar_documento(idLegado):

    if not os.path.exists(nltk.data.find('corpora/stopwords')):
        nltk.download('stopwords')

    if not os.path.exists(nltk.data.find('tokenizers/punkt')):
        nltk.download('punkt')

    print("inciado a classificação")  

    conn = msql.connect(host='localhost',
                    database='classificador',
                    user='root',
                    password='root')

    cursor = conn.cursor()  

    nIdLegado = idLegado[0][0]
    cFileName = idLegado[0][1]

    parte_do_nome = cFileName

    diretorio = r'C:\\_diogenes_\\LEXi\\AtasRead_legado\\melhoria_textos\\api\\tmp\\'

    pdf_path_original = localizar_arquivos_por_parte_do_nome(parte_do_nome)   

    print(pdf_path_original)
    
    diretorioIMG = r'C:\\_diogenes_\\LEXi\\AtasRead_legado\\melhoria_textos\\api\\img\\'

    pdf_path = pdf_path_original #.replace("\\", "\\\\")

    texto_extraido = ""
 

    lRetProcessamento = False

    if pdf_path != " ":  

        #texto_extraido = extrair_texto_do_pdf(pdf_path)
        cTextoPronto = ""

        if len(texto_extraido) <= 0:

            extensao = os.path.splitext(pdf_path)[1]                

            if extensao == '.pdf':

                try:
                    print("pdf_path: ",pdf_path)
                    print("diretorio: ", pdf_path)

                    pdf_to_jpg(pdf_path, diretorioIMG)                        
                    arquivos_jpg = [arquivo for arquivo in os.listdir(diretorioIMG) if arquivo.endswith('.jpg')]      
                    print(arquivos_jpg)    
                    
                    if len(arquivos_jpg) > 0:

                        pytesseract.pytesseract.tesseract_cmd =  r'C:\\_diogenes_\\LEXi\\Tesseract\\tesseract.exe'
                        tessdata_dir_config = r'--tessdata-dir "C:\\_diogenes_\\LEXi\\Tesseract\\tessdata\\"'

                        custom_config = r'--oem 3 --psm 6'

                        for arquivo in arquivos_jpg:    
                                                        
                            caminho_completo = os.path.join(diretorioIMG, arquivo)                                                            
                            imagem = Image.open(caminho_completo)
                            imagem = imagem.convert("L")
                            nova_resolucao = (imagem.width * 2, imagem.height * 2)
                            imagem = imagem.resize(nova_resolucao, Image.BILINEAR)
                            contraste = ImageEnhance.Contrast(imagem)
                            imagem = contraste.enhance(1.5)  # Ajustando o contraste em 50%
                            iluminacao = ImageEnhance.Brightness(imagem)
                            imagem = iluminacao.enhance(1.2)  # Aumentando a iluminação em 20%    
                                                    
                            texto = pytesseract.image_to_string(imagem, lang='por')
                                
                                
                            texto_extraido += texto
                                
                            os.remove(caminho_completo)

                    cTextoPronto = limpar_texto(texto_extraido)

                except:
                    update_query = "update classificador.legado set statusimport='ERROR', textfull='ERRO: Problema na gravação dos dados.', pathdoc=%s where idLegado = %s "
                    cursor.execute(update_query, (pdf_path_original, str(cFileName)))

                    if cursor.rowcount > 0:
                        print("Execute query SUCESSO - OK 4 - ID: "+str(nIdLegado))
                        conn.commit()
                    else:
                        print("Execute query FALHA - OK 4 - ID: "+str(nIdLegado))  

            elif extensao == '.html':

                    # Detectar a codificação do arquivo
                    with open(pdf_path, 'rb') as arquivo:
                        # Lendo os primeiros 10.000 bytes para detectar a codificação
                        dados = arquivo.read(10000)
                        resultado = chardet.detect(dados)

                    # Obtendo a codificação detectada
                    codificacao = resultado['encoding']

                    # Abrir o arquivo com a codificação detectada
                    with open(pdf_path, 'r', encoding=codificacao) as arquivo:
                        conteudo = arquivo.read()

                        # Analisando o HTML com BeautifulSoup
                        soup = BeautifulSoup(conteudo, 'html.parser')

                        # Extraindo texto do HTML
                        texto = soup.get_text()

                    texto_extraido = texto

            cTextoPronto = limpar_texto(texto_extraido)

            if len(cTextoPronto) > 0:
                cursor = conn.cursor()

                try:

                    update_query = "update classificador.legado set statusimport='OK', textfull=%s, pathdoc=%s where idLegado = %s "
                    cursor.execute(update_query, (cTextoPronto, pdf_path_original, str(nIdLegado)))

                    if cursor.rowcount > 0:
                        # print("Execute query SUCESSO - OK 1 ")
                        __result = "Execute query SUCESSO - OK 1 - ID: " +   str(nIdLegado)
                        lRetProcessamento=True
                        conn.commit()
                    else:
                        # print("Execute query FALHA 1")
                        __result = "Execute query FALHA - OK 1 - ID: "+str(nIdLegado)
                except:
                    update_query = "update classificador.legado set statusimport='ERROR', textfull='ERRO: Problema na gravação dos dados.', pathdoc=%s where idLegado = %s "
                    cursor.execute(update_query, (pdf_path_original,  str(nIdLegado)))

                    if cursor.rowcount > 0:
                        __result = "Execute query SUCESSO - OK 2 - ID: " + str(nIdLegado)
                        conn.commit()
                    else:
                        __result = "Execute query FALHA - OK 2 - ID: "+str(nIdLegado)

            else:
            
                update_query = "update classificador.legado set statusimport='ERROR', textfull='ERRO: Problema na gravação dos dados.', pathdoc=%s where idLegado = %s "
               # print(update_query, (pdf_path_original, str(cFileName)))
                cursor.execute(update_query, (pdf_path_original, str(nIdLegado)))
                

                if cursor.rowcount > 0:
                    __result = "Execute query SUCESSO - OK 3 - ID: "+str(nIdLegado)
                    conn.commit()
                else:
                    __result = "Execute query FALHA - OK 3 - ID: "+str(nIdLegado)

        else:
            print("ERRO: no array de imagem.")
            update_query = "update classificador.legado set statusimport='ERROR', textfull='ERRO: Problema na gravação dos dados.', pathdoc=%s where idLegado = %s "
            cursor.execute(update_query, (pdf_path_original, str(idLegado)))

            if cursor.rowcount > 0:
                __result = "Execute query SUCESSO - OK 4 - ID: "+str(nIdLegado)
                conn.commit()
            else:
                __result = "Execute query FALHA - OK 4 - ID: "+str(nIdLegado)
    

 
    return(lRetProcessamento)


# Restante do seu código aqui...

# if __name__=='__main__':

#     try:
        
#         if not os.path.exists(nltk.data.find('corpora/stopwords')):
#             nltk.download('stopwords')

#         if not os.path.exists(nltk.data.find('tokenizers/punkt')):
#             nltk.download('punkt')
#                                                                        #  WHERE statusimport is null statusimport='ERROR'                   
#         query = "SELECT iddocumento, idLegado FROM classificador.legado  Where statusimport is null  "
#         cursor.execute(query)
#         result = cursor.fetchall()
                
#         # Obtém o número de núcleos do processador
#         num_cores = multiprocessing.cpu_count()
#         result_process = Parallel(n_jobs=12, backend="multiprocessing")(delayed(processar_documento)(n) for n in result)
#         print(result_process)

#         cursor.close()
#         conn.close()

#         print("Importação concluída com sucesso.")

#     except Exception as e:
#         # emitir_som('Alarm09.wav')
#         print("ERROR....")
