from flask import Flask, request, render_template, redirect, url_for
from flask_cors import CORS, cross_origin
import numpy as np
app = Flask(__name__)

cors = CORS(app, resources={r"*": {"origins": "*"}})

@app.route('/', methods=['GET','POST'])
def home():
    if request.headers['Finished']:
        print("Final Wav called!")
        data = np.array(request.get_data())
        np.save(data,'saved_numpy')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
