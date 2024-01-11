import pandas as pd
import mysql.connector as msql
import PyPDF2
import pytesseract
from PIL import Image
from PIL import ImageEnhance
import io
import os
import PyPDF2
from PyPDF2 import PdfFileReader
from pdf2image import convert_from_path
import os

import re
import nltk
import unidecode
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from unicodedata import normalize
import pygame


def localizar_arquivos_por_parte_do_nome(parte_do_nome):
    diretorio = 'H:\\_diogenes_\\queiroz&cavalcante\\ged_queiroz\\' 

    caminho_completo=''
    for pasta_raiz, subpastas, arquivos in os.walk(diretorio):
        for arquivo in arquivos:
            if parte_do_nome in arquivo:
                caminho_completo = os.path.join(pasta_raiz, arquivo)
              
    return(caminho_completo)
              

def pdf_to_jpg(pdf_file, output_folder):
    
    try:

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        pdf = PdfFileReader(open(pdf_file, 'rb'))
        num_pages = pdf.getNumPages()

        for page_num in range(num_pages):
            images = convert_from_path(
                pdf_file,
                first_page=page_num + 1,
                last_page=page_num + 1,
                poppler_path=r'H:\\_diogenes_\\ClassificadorAtas\\poppler-23.11.0\\Library\\bin'
            )

            for i, image in enumerate(images):
                image.save(os.path.join(output_folder, f'page_{page_num + 1}_{i + 1}.jpg'), 'JPEG')
    except:
        print("ERRO: Arquivo no formato diferete ou corrompto.")
            

def extrair_texto_do_pdf(pdf_path):
    texto = ""
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            texto += page.extract_text()
    print("extraio por texto")
    return texto


def remover_acentos(txt):
    return normalize('NFKD', txt).encode('ASCII', 'ignore').decode('ASCII')


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
    #texto = remover_acentos(texto)
    #texto = remove_punctuation(texto)
    
    # Remover caracteres especiais e pontuações, incluindo acentos
    texto = unidecode.unidecode(texto)
    texto = re.sub(r'[^\w\s]', ' ', texto)  # Modificando a expressão regular para manter números
    
    # Tokenizar o texto em palavras
    palavras = word_tokenize(texto)
    
    # Remover stopwords em português
    stopwords_portugues = set(stopwords.words('portuguese'))
    palavras_sem_stopwords = [palavra for palavra in palavras if palavra not in stopwords_portugues]
    
    # Juntar as palavras novamente em um texto limpo
    texto_limpo = ' '.join(palavras_sem_stopwords)
    
    return texto_limpo


def emitir_som(caminho_som):
    pygame.init()
    pygame.mixer.init()
    try:
        som = pygame.mixer.Sound(caminho_som)
        som.play()
        while pygame.mixer.get_busy():
            pygame.time.delay(100)
    except pygame.error as e:
        print(f"Ocorreu um erro: {e}")


try: 
    nltk.download('stopwords')
    nltk.download('punkt')

    pytesseract.pytesseract.tesseract_cmd =  r'H:\\_diogenes_\\ClassificadorAtas\\Tesseract\\tesseract.exe'
    tessdata_dir_config = r'--tessdata-dir "H:\\_diogenes_\\ClassificadorAtas\\Tesseract\\tessdata"'
    
    custom_config = r'--oem 3 --psm 6'  # Definindo o modo de reconhecimento e segmentação de página

    diretorio = r'H:\\_diogenes_\\LEXi\AtasRead_legado\\melhoria_textos\\temp\\'

    conn = msql.connect(host='localhost',
                            database='classificador', 
                            user='root',
                            password='123456')
    cursor = conn.cursor()

    if conn.is_connected():        
            cTextoPronto=' '
            output_f = 'temp'

            query = "SELECT iddocumento, idLegado FROM classificador.legado WHERE statusimport is null"
            cursor.execute(query)
            result = cursor.fetchall()

            for n in result:    
                parte_do_nome = str(n[0])
                print("Processando ID Doc: ", n[0])
                
                pdf_path_original = localizar_arquivos_por_parte_do_nome(parte_do_nome)
                pdf_path = pdf_path_original.replace("\\", "\\\\")    
                
                if pdf_path != " ": 
                    texto_extraido = ""
                                    
                    #texto_extraido = extrair_texto_do_pdf(pdf_path)

                    if len(texto_extraido)<=0:
                            
                        pdf_to_jpg(pdf_path, output_f)     
                            
                        arquivos_jpg = [arquivo for arquivo in os.listdir(diretorio) if arquivo.endswith('.jpg')]
                                        
                        if len(arquivos_jpg)>0:
                        
                            # Itera sobre os arquivos e extrai o texto
                            for arquivo in arquivos_jpg:
                                caminho_completo = os.path.join(diretorio, arquivo)
                                imagem = Image.open(caminho_completo)
                                imagem = imagem.convert("L")                            
                                nova_resolucao = (imagem.width * 2, imagem.height * 2)  # Dobrando a resolução
                                imagem = imagem.resize(nova_resolucao, Image.BILINEAR)  # Pode-se usar outros métodos de interpolação                         
                                contraste = ImageEnhance.Contrast(imagem)
                                imagem = contraste.enhance(1.5)  # Ajustando o contraste em 50%                           
                                iluminacao = ImageEnhance.Brightness(imagem)
                                imagem = iluminacao.enhance(1.2)  # Aumentando a iluminação em 20%
                                texto = pytesseract.image_to_string(imagem,lang='por') #, lang='por', config=tessdata_dir_config+custom_config
                                texto_extraido += texto

                                os.remove(caminho_completo)

                    cTextoPronto = limpar_texto(texto_extraido) 
                    
                    if len(cTextoPronto) >0 :
                        cursor = conn.cursor()
                        
                        try:
                            
                            update_query = "update classificador.legado set statusimport='OK', textfull=%s, pathdoc=%s where idLegado = %s "
                            cursor.execute(update_query, (cTextoPronto, pdf_path_original ,str(n[1])))

                            if cursor.rowcount > 0:
                                print("Execute query SUCESSO - OK 1 ")
                                conn.commit()
                            else:
                                print("Execute query FALHA 1")     
                        except:
                            update_query = "update classificador.legado set statusimport='ERROR', textfull='ERRO: Problema na gravação dos dados.', pathdoc=%s where idLegado = %s "
                            cursor.execute(update_query, ( pdf_path_original,str(n[1])))

                            if cursor.rowcount > 0:
                                print("Execute query SUCESSO - OK 2 ")
                                conn.commit()
                            else:
                                print("Execute query FALHA 2 ")
                                            
                        
                    else:
                        update_query = "update classificador.legado set statusimport='ERROR', textfull='ERRO: Problema na gravação dos dados.', pathdoc=%s where idLegado = %s "
                        cursor.execute(update_query, (pdf_path_original,str(n[1])))

                        if cursor.rowcount > 0:
                            print("Execute query SUCESSO - OK 3 ")
                            conn.commit()
                        else:
                            print("Execute query FALHA 3 ")
                        
                else:
                    print("ERRO: no array de imagem.")
                    update_query = "update classificador.legado set statusimport='ERROR', textfull='ERRO: Problema na gravação dos dados.', pathdoc=%s where idLegado = %s "
                    cursor.execute(update_query, (pdf_path_original,str(n[1])))

                    if cursor.rowcount > 0:
                        print("Execute query SUCESSO - OK 4 ")
                        conn.commit()
                    else:
                        print("Execute query FALHA 4 ")
            
    else:
        print("erro de conexão")

    cursor.close()
    conn.close()

    print("Importação concluída com sucesso.")
except:
    emitir_som('Alarm09.wav')
