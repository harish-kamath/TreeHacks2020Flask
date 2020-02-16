from flask import Flask, request, render_template, redirect, url_for
from flask_cors import CORS, cross_origin
import numpy as np
import scipy.io.wavfile as wavf
from processWav import getAttribs
app = Flask(__name__)

cors = CORS(app, resources={r"*": {"origins": "*"}})

current_partial = None

@app.route('/', methods=['GET','POST'])
def home():
    global current_partial
    if request.headers['Finished'] == 'true':
        print("Final Wav called!")
        wavf.write('final.wav',44100,np.array([float(x) for x in current_partial.decode('utf-8').split(',')]))
        current_partial = None
    else:
        if current_partial == None:
            current_partial = request.get_data()
        else:
            current_partial += b','
            current_partial += request.get_data()
        wavf.write('temp.wav',44100,np.array([float(x) for x in current_partial.decode('utf-8').split(',')]))
        data = getAttribs("temp.wav")
        # Pitch, Jitter, Shimmer
        return str(data[0]) + "," + str(data[3]) + "," + str(data[8])
    return "Hello, World!"
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
