from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/receive-ok', methods=['POST'])
def receive_ok():
    try:
        data = request.get_json()

        if data is None or 'word' not in data or data['word'] != 'ok':
            return jsonify({'message': 'Palavra "ok" não encontrada no JSON enviado'}), 400

        return jsonify({'message': 'Recebido'}), 200
    
    except Exception as e:
        return jsonify({'message': f'Erro ao processar a solicitação: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
