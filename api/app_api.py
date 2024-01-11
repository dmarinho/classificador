from flask import Flask, request, jsonify
import os
import mysql.connector as msql
import Proc_Novo_arquivo
import Classifica_Doc

app = Flask(__name__)

@app.route('/receive-pdf', methods=['POST'])
def receive_pdf():  

    conn = msql.connect(host='localhost',
            database='classificador', 
            user='root',
            password='root')
    cursor = conn.cursor()
# try:
    if 'file' not in request.files:
        return jsonify({'message': 'Nenhum arquivo enviado'}), 400

    file = request.files['file']
    if file.filename == '' or not file.filename.endswith('.pdf'):
        return jsonify({'message': 'Por favor, envie um arquivo PDF'}), 400

    query = "SELECT idLegado FROM classificador.legado  Where nomedoc= '"+file.filename+"' "
    cursor.execute(query)
    result = cursor.fetchall()
    cClassificado=""

    if len(result)==0:           
        save_path = r'C:\\_diogenes_\\LEXi\\AtasRead_legado\\melhoria_textos\\api\\tmp\\'  # Pasta de destino para salvar o arquivo
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file.save(os.path.join(save_path, file.filename))

        cursor = conn.cursor()

        insert_query = "INSERT INTO `classificador`.`legado` (iddocumento, npu, textfull, classificado, nomedoc, textonormalizado, pathdoc, statusimport, dthrclassificado, resultclassificado, novadtaudiencia, novahraudiencia) VALUES (%s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s)"
        cursor.execute(insert_query, ('','','','',file.filename,'','Importado Via API','','','','',''))
        if cursor.rowcount > 0:
           #print("Execute SUCESSO")
            conn.commit()
                                                                    
            query = "SELECT idLegado, nomedoc FROM classificador.legado  Where nomedoc= '"+file.filename+"' "
            cursor.execute(query)
            result = cursor.fetchall()
            nIdLegado = result[0][0]
            lRet = Proc_Novo_arquivo.processar_documento(result)     
            if lRet==True:
                cClassificado = Classifica_Doc.classifica_nova_ata(nIdLegado)
            
            if cClassificado !='':
                return jsonify({'message':cClassificado }), 200
            
        else:
            print("Execute FALHA")
    else:
       return jsonify({'message': "Arquivo já existe na base"}), 200

    return jsonify({'message': "Recebido"}), 200

  #  except Exception as e:
   #     return jsonify({'message': f'Erro ao receber o arquivo: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='192.168.2.13', port='5000')
