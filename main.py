from flask import Flask, request, render_template, redirect, url_for
from flask_cors import CORS, cross_origin
app = Flask(__name__)

cors = CORS(app, resources={r"/houndifyAuth": {"origins": "http://localhost:3446"},r"/textSearchProxy": {"origins": "http://localhost:3446"}})

@app.route('/')
def home():
    return render_template('/Hound/public/index.html')

@app.route('/file-upload', methods=['POST'])
def file_upload():
    print("OMGMGGGGGG")
    app.logger.debug(request.files)
    return render_template('/Hound/public/index.html')

@app.route('/houndifyAuth')
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def houndify_auth():
    print(request)
    return redirect('http://localhost:3446/houndifyAuth?token='+str(request.args.get('token')), code=301)

@app.route('/textSearchProxy')
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def text_search_proxy():
    print(request)
    return redirect('http://localhost:3446/textSearchProxy?token='+str(request.args.get('token')), code=301)


if __name__ == '__main__':
    app.run()
