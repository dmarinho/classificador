from flask import Flask

UPLOAD_FOLDER = r'C:\\_diogenes_\\LEXi\AtasRead_legado\\melhoria_textos\\api\\tmp'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024