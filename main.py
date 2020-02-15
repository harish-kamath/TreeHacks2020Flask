from flask import Flask, request, render_template, redirect, url_for
from flask_cors import CORS, cross_origin
app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, World!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
