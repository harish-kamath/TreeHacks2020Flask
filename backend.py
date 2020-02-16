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
        arr = np.array([float(x) for x in request.get_data().decode('utf-8').split(',')])
        arr = (arr*32767).astype(np.int16)
        wavf.write(request.headers['Key_Current']+'_'+request.headers['Test']+'.wav',44100,arr)
        data = getAttribs(request.headers['Key_Current']+'_'+request.headers['Test']+'.wav')
        # Pitch, Jitter, Shimmer
        return str(data[0]) + "," + str(data[3]) + "," + str(data[8])
    return "Hello, World!"
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
