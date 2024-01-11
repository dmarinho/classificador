from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Rota para receber o PDF
@app.route('/receber_pdf', methods=['POST'])
def receber_pdf():
    if 'pdf' not in request.files:
       return jsonify({'error': 'erro....'}), 400

    pdf_file = request.files['pdf']
    
    if pdf_file.filename == '':
        return jsonify({'error': 'Nome de arquivo PDF inválido'}), 400

    if pdf_file:
        # Salvar o PDF na pasta TEMP
        temp_folder = 'TEMP'
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
        
        pdf_path = os.path.join(temp_folder, pdf_file.filename)
        pdf_file.save(pdf_path)

        return jsonify({'message': f'PDF recebido e salvo em {pdf_path}'}), 200

if __name__ == '__main__':
    app.run(debug=False)
