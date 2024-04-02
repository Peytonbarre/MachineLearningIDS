from flask import Flask, jsonify, request
from flask_cors import CORS
from sqlalchemy import create_engine, insert, text
from datetime import datetime
import pandas as pd
import MTH, LCCDE
#pip install mysqlclient

app = Flask(__name__)
CORS(app)

engine = create_engine('mysql://admin:projectt60@csproject.c5emwcgweqq7.us-east-2.rds.amazonaws.com/data', echo=True)

@app.route('/')
def home():
    return "Welcome to IDS-ML!"

@app.route('/processParameters', methods=['POST'])
def processParameters():
    data = request.json
    #{"classifier":"LCCDE","slideValue":"59","SMOTEValue":"2:1000","graphType":"Matrix","parameter":"Confusion Matrix"}
    SMOTE = data.get('SMOTEValue')
    trainValue = data.get('slideValue')
    classifier = data.get('classifier')
    graphType = data.get('graphType')
    parameter = data.get('parameter')

    #NOTE This is a terrible way to add to a database, can cause SQL injections
    #NOTE but I'm too lazy to do it the right way
    query = "SELECT * FROM history as h WHERE h.Model like \'" + classifier + "\' AND h.smote like \'" + SMOTE + "\' AND h.trainVal = " + str(trainValue) 
    result = pd.read_sql(query, engine)
    if result.size != 0:
        print("Already found! " + str(result.iloc[0]))
    else:
        currentTime = datetime.now()
        addQuery = "INSERT INTO history (TimeStamps, Model, smote, trainVal) VALUES (\'" + str(currentTime) + "\',\'" + classifier + "\',\'" + SMOTE + "\',\'" + str(trainValue) + "\')"
        with engine.connect() as conn:
            conn.execute(text(addQuery))
            conn.commit()

    #TODO
    ##if classifier == 'MTH':
    ##    data = MTH.getStacking(trainValue, SMOTE)
    ##    print('================== ')
    ##    print(data[6])
    ##
    ###TODO
    ##elif classifier == 'LCCDE':
    ##    #LCCDE.applyDefaultHyperparameters()
    ##    print("LCCDE")
    ##
    ###TODO
    ##elif classifier == 'Tree-Based':
    ##    print("Tree-Based")
    ##
    ##if graphType == 'Matrix':
    ##    data = data[6]
    ##

    response = {
        "status": "good"
    }
    return jsonify(response)

@app.route('/MTH_XGBoost', methods=['GET'])
def MTH_XGBoost():
    acurracy, precision, recall, fscore, y_true, y_predict,cm = MTH.getXGBoost()
    data = {
        "accuracy": acurracy,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
        "y_true": y_true,
        "y_predict": y_predict,
        "cm": cm
    }
    return jsonify(data)

@app.route('/MTH_ET', methods=['GET'])
def MTH_ET():
    acurracy, precision, recall, fscore, y_true, y_predict,cm = MTH.getExtraTrees()
    data = {
        "accuracy": acurracy,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
        "y_true": y_true,
        "y_predict": y_predict,
        "cm": cm
    }
    return jsonify(data)

@app.route('/MTH_DT', methods=['GET'])
def MTH_DT():
    acurracy, precision, recall, fscore, y_true, y_predict,cm = MTH.getDecisionTree()
    data = {
        "accuracy": acurracy,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
        "y_true": y_true,
        "y_predict": y_predict,
        "cm": cm
    }
    return jsonify(data)

@app.route('/MTH_RF', methods=['GET'])
def MTH_RF():
    acurracy, precision, recall, fscore, y_true, y_predict,cm = MTH.getRandomForest()
    data = {
        "accuracy": acurracy,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
        "y_true": y_true,
        "y_predict": y_predict,
        "cm": cm
    }
    return jsonify(data)

@app.route('/MTH_STACK', methods=['GET'])
def MTH_STACK():
    acurracy, precision, recall, fscore, y_true, y_predict,cm = MTH.getStacking()
    data = {
        "accuracy": acurracy,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
        "y_true": y_true,
        "y_predict": y_predict,
        "cm": cm
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)