# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 22:49:59 2020

@author: Shrita
"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('sc.pkl','rb'))
pca = pickle.load(open('pca.pkl','rb'))
le = pickle.load(open('le.pkl','rb'))

@app.route('/')

def home():
    return render_template('untitled1.html')

@app.route('/predict',methods = ['POST'])

def predict():
    int_features = [x for x in request.form.values() ]
    option = request.form['options']
    option1 = request.form['options1']
    option2 = request.form['options2']
    if option == 'Yes':
        int_features[0] = 1
    else:
        int_features[0] = 0
    if option1 == 'Yes':
        int_features[7] = 1
    else:
        int_features[7] = 0
    if option2 == 'Yes':
        int_features[5] = 1
    else:
        int_features[5] = 0
    final_features = [float(x) for x in int_features]
    final_features = [np.array(int_features)]
    final_features = sc.transform(final_features)
    final_features = pca.transform(final_features)
    prediction = model.predict(final_features)
    if prediction == 1:
        out = 'SUCCEED'
    else:
        out  = 'FAIL'
    
    return render_template('untitled1.html',prediction_text = '**There is a greater probability for the startup to {}, according to the inputs**'.format(out))

if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    
    
    
    