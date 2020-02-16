from flask import Flask, request, render_template, redirect, url_for
from flask_cors import CORS, cross_origin
import numpy as np
import scipy.io.wavfile as wavf
from processWav import getAttribs
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import mir_eval
import librosa
from sklearn.externals import joblib
from keras.models import model_from_json
import myspsolution as mysp
import nn
app = Flask(__name__)

cors = CORS(app, resources={r"*": {"origins": "*"}})

current_partial = None
current_id = 0

emotions = ['Angry','Calm','Fearful','Happy','Sad','Angry','Calm','Fearful','Happy','Sad']

def gender_age(path):
    clffg = joblib.load('cfl_gender.pkl')
    clffa = joblib.load('cfl_age.pkl')
    y, sr = librosa.load(path, sr=22050)
    spectrogram = np.abs(librosa.stft(y))
    melspec = librosa.feature.melspectrogram(y=y, sr=sr)
    stft = np.abs(librosa.stft(y))
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
    features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
    features = features.reshape(1, -1)
    gend_pred = clffg.predict(features)
    print("Gender Prediction: ", gend_pred)
    age_pred = clffa.predict(features)
    print("Age Prediction: ", age_pred)
    return gend_pred,age_pred

def emotion(path):
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("Emotion_Voice_Detection_Model.h5")
    X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
    featurelive = mfccs
    livedf2 = featurelive
    livedf2= pd.DataFrame(data=livedf2)
    livedf2 = livedf2.stack().to_frame().T
    twodim= np.expand_dims(livedf2, axis=2)
    livepreds = loaded_model.predict(twodim, batch_size=32, verbose=0)
    livepreds1=livepreds.argmax(axis=1)
    liveabc = livepreds1.astype(int).flatten()
    full = sum(livepreds[0])
    emotion_scores = {k:0 for k in emotions}
    for i in range(len(livepreds[0])):
        emotion_scores[emotions[i]] += livepreds[0][i]/full * 100
    print(emotions[liveabc[0]])
    print(emotion_scores)
    return emotions[liveabc[0]], emotion_scores


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

@app.route('/report/<reportid>')
def report(reportid):
    current_id = reportid
    print('Showing Report '+str(reportid))
    files = [x for x in os.listdir() if (str(reportid) in x and "wav" in x)]
    print(files)
    for path in files:
        data = getAttribs(path)
        pitch = data[0]
        jitter = data[1]
        shimmer = data[2]

        gender, age = gender_age(path)
        current_emotion, emotion_scores = emotion(path)
        ts = 0
        for a in emotion_scores.keys():
            ts += emotion_scores[a]
        for a in emotion_scores.keys():
            emotion_scores[a] = abs(emotion_scores[a]+np.random.randn()*6 - 3)/ts * 100
        total_values = mysp.mysptotal(path.replace(".wav",""),os.getcwd())
        total_values = total_values.to_dict()
        for k in total_values.keys():
            total_values[k] = float("{0:.3f}".format(float(total_values[k][0])))
        if(str(current_id)+'_alexarecord.txt' in os.listdir()):
            f = open(str(current_id)+'_alexarecord.txt', 'r')
            total_values['selfdiag'] = f.readlines()[0]
            f.close()

        total_values['gender'] = gender[0]
        total_values['age'] = age[0]
        total_values['current_emotion'] = current_emotion
        total_values['emotion_scores'] = emotion_scores
        total_values['pitch'] = pitch
        total_values['jitter'] = jitter
        total_values['shimmer'] = shimmer
        total_values['parkinson'] = nn.percentChance(path)*100*0.4
        return render_template('pages/blank-page.html', values=total_values)

@app.route('/alexarecord', methods=['POST'])
def alexapost():
    f = open(str(current_id)+'_alexarecord.txt','w+')
    f.write(request.headers['Problem'])
    f.close()
    return "Hello, World!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
