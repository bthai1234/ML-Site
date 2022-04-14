from distutils.version import Version
from urllib import response
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
import pandas as pd
import csv
import json

import numpy as np
from sklearn import svm
from sklearn import datasets
import joblib
from sklearn.model_selection import train_test_split

# declare constants
HOST = '0.0.0.0'
PORT = 8081

# initialize flask application
app = Flask(__name__)
CORS(app)
app.config["DEBUG"] = True

@app.route('/api/train', methods=['POST'])
def train():
    # get parameters from request
    parameters = request.get_json()
    version = float(parameters['version'])
    # read iris data set
    iris = pd.read_csv("./Data sets/iris.csv") 
    X, y = iris.drop(columns="variety"), iris["variety"]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

    # fit model
    clf = svm.SVC(C=float(parameters['C']),
                  probability=True,
                  random_state=1)
    clf.fit(X_train, y_train)
    # persist model
    joblib.dump(clf, './Models/iris_model_{version}.pkl'.format(version=version))
    return jsonify({'accuracy': round(clf.score(X_test, y_test) * 100, 2)})

@app.route('/iris/svm/predict', methods=['POST'])
def irisPredict():
    # get parameters from request
    inputData = request.get_json()
    # get iris object from request
    features = [[float(inputData['sepal_length']), 
                float(inputData['sepal_width']), 
                float(inputData['petal_length']), 
                float(inputData['petal_width'])]]

    version = int(inputData['version'])
    
    # read iris data set
    clf = joblib.load('./Models/iris_model_{version}.pkl'.format(version=version))
    probabilities = clf.predict_proba(features)

    return jsonify([
        {'name': 'Iris-Setosa', 'value': round(probabilities[0, 0] * 100, 2)},
        {'name': 'Iris-Versicolour', 'value': round(probabilities[0, 1] * 100, 2)},
        {'name': 'Iris-Virginica', 'value': round(probabilities[0, 2] * 100, 2)},
        ])

@app.route('/iris/getData', methods=['GET'])
def getData():
    #data = open("./Data sets/iris.csv", "r")
    #reader = csv.DictReader(data)
    #out = json.dumps([ row for row in reader ])
    iris = pd.read_csv("./Data sets/iris.csv")
    #iris = iris.head(n=50)
    iris = iris.to_json(orient="records")

    print("Iris data sent!")  
    return iris

@app.route('/iris/getTypes', methods=['GET'])
def getTypes():
    iris = pd.read_csv("./Data sets/iris.csv")
    getTypesPercent = iris['variety'].value_counts(normalize=True) * 100
    getTypesPercent = getTypesPercent.reset_index()
    getTypesPercent.columns = ['variety', 'percent']
    getTypesPercent = getTypesPercent.to_json(orient="records")
    print("Iris data Types percent sent!")  
    return getTypesPercent

if __name__ == '__main__':
    # run web server
    app.run(host=HOST,
            debug=True,  # automatic reloading enabled
            port=PORT)
