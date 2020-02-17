from flask import Flask, request, render_template, redirect, url_for
from flask_cors import CORS, cross_origin
from processWav import getAttribs

app = Flask(__name__)

cors = CORS(app, resources={r"*": {"origins": "*"}})

@app.route('/')
def home():
    return "Hello, World!"
    
@app.route('/fileProcess/')
def fileProcess():
    data = getAttribs("morph1Female.wav")
    return "Pitch: " + str(data[0]) + "\nJitter: " + str(data[3]) + "\nShimmer: " + str(data[8])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=50)
