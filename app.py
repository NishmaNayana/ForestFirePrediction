# -*- coding: utf-8 -*-
"""
Created on Sat May 22 21:01:52 2021

@author: ASUS
"""
from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
import joblib
import numpy as np

model=pickle.load(open('etrmodel.pkl','rb'))


app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def result():
    '''d1=request.form['month']
    d2=request.form['day']
    d3=request.form['x']
    d4=request.form['y']
    d5=request.form['ffmc']'''
    
    d6=request.form['dmc']
    d7=request.form['dc']
    '''d8=request.form['isi']'''
    
    d9=request.form['temp']
    d10=request.form['rh']
    d11=request.form['wind']
    '''d12=request.form['rain']'''
    arr=np.array([[d6,d7,d9,d10,d11]])
    pred=model.predict(arr)
    output=round(pred[0],2)
    return render_template('index.html',pred_text='Burnt area is : {} ha'.format(output))
    
  
    
if __name__=='__main__':
    app.run(debug=True)
