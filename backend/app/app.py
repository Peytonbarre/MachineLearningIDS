import os
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from sqlalchemy import create_engine, text
from datetime import datetime
import pandas as pd
import MTH, LCCDE, TreeBased
#pip install mysqlclient

app = Flask(__name__)
CORS(app)
load_dotenv()

#'mysql://admin:PASSWORD@csproject.c5emwcgweqq7.us-east-2.rds.amazonaws.com/data'
DATABASE_URL = os.getenv('DATABASE_URL')
print(DATABASE_URL)
engine = create_engine(DATABASE_URL, echo=True)

@app.route('/')
def home():
    return "Welcome to IDS-ML!"

@app.route('/processParameters', methods=['POST'])
def processParameters():
    data = request.json
    #{"classifier":"LCCDE","slideValue":"59","SMOTEValue":"2:1000","graphType":"Matrix","parameter":"Confusion Matrix"}
    
    SMOTE = data.get('SMOTEValue')
    trainValue = str(round(float(data.get('slideValue'))/100,2))
    classifier = data.get('classifier')
    graphType = data.get('graphType')
    parameter = data.get('parameter')
  
    #TODO When done with metric derivation, make this SQL statement add ALL metrics for each query (move below)
    # Use parameterized query to prevent SQL injection
    query = text("SELECT * FROM history WHERE Model = :classifier AND smote = :SMOTE AND trainVal = :trainValue")
    result = pd.read_sql(query, engine, params={"classifier": classifier, "SMOTE": SMOTE, "trainValue": trainValue}) 

    if not result.empty:
        print("Already found! " + str(result.iloc[0]))
        #TODO
        data
    else:
        #Data so far: acurracy, precision, recall, fscore, y_true, y_predict, cm
        #TODO: Make sure this is working
        if classifier == 'MTH':
            data = MTH.getStacking(trainValue, SMOTE)
        #TODO: Make sure this is working
        elif classifier == 'LCCDE':
            data = LCCDE.applyDefaultHyperparameters(trainValue, SMOTE)
        #TODO: Make sure this is working
        elif classifier == 'Tree-Based':
            data = TreeBased.applyDefaultHyperparameters(trainValue, SMOTE)
    
    #TODO: Add derivation for these parameters in the classifier files
    #Confusion Matrix       [*]
    #Average of Event       [*]
    #Prevision by Event     [*]
    #Recall by Event        [*]
    #F1 by Event            [*]
    #Accuracy by Event      [*]
    #Classifier Composition [ ]
    #Average Accuracy       [ ]
    #Average F1 Score       [ ]

    if graphType == 'Matrix':
        data = data[6]
    elif graphType == 'Line':
        data = data[7]
    elif graphType == 'Bar':
        if parameter == 'Precision by Event':
            data = data[8]
        elif parameter == 'Recall by Event':
            data = data[10]
        elif parameter == 'F1 by Event':
            data = data[9]
        elif parameter == 'Accuracy by Event':
            data = data[11]
    elif graphType == 'Pie':
        #TODO: Change to classifier composition
        data = data[6]
    elif graphType == 'Callout':
        if parameter == 'Avg Accuracy':
            data = data[0]
        elif parameter == 'Avg Precision':
            data = data[0]
        elif parameter == 'Avg Recall':
            data = data[0]
        elif parameter == 'Avg F1':
            data = data[0]

    response = {
        "status": "good",
        "data": data
    }

    currentTime = datetime.now()
    addQuery = text("INSERT INTO history (TimeStamps, Model, smote, trainVal) VALUES (:currentTime, :classifier, :SMOTE, :trainValue)")
    with engine.connect() as conn:
        conn.execute(addQuery, {"currentTime": currentTime, "classifier": classifier, "SMOTE": SMOTE, "trainValue": trainValue})
        conn.commit()

    return jsonify(response)


##
## Code below is depricated, we probably won't use this but leaving it here just in case
##

# @app.route('/MTH_XGBoost', methods=['GET'])
# def MTH_XGBoost():
#     acurracy, precision, recall, fscore, y_true, y_predict,cm = MTH.getXGBoost()
#     data = {
#         "accuracy": acurracy,
#         "precision": precision,
#         "recall": recall,
#         "fscore": fscore,
#         "y_true": y_true,
#         "y_predict": y_predict,
#         "cm": cm
#     }
#     return jsonify(data)

# @app.route('/MTH_ET', methods=['GET'])
# def MTH_ET():
#     acurracy, precision, recall, fscore, y_true, y_predict,cm = MTH.getExtraTrees()
#     data = {
#         "accuracy": acurracy,
#         "precision": precision,
#         "recall": recall,
#         "fscore": fscore,
#         "y_true": y_true,
#         "y_predict": y_predict,
#         "cm": cm
#     }
#     return jsonify(data)

# @app.route('/MTH_DT', methods=['GET'])
# def MTH_DT():
#     acurracy, precision, recall, fscore, y_true, y_predict,cm = MTH.getDecisionTree()
#     data = {
#         "accuracy": acurracy,
#         "precision": precision,
#         "recall": recall,
#         "fscore": fscore,
#         "y_true": y_true,
#         "y_predict": y_predict,
#         "cm": cm
#     }
#     return jsonify(data)

# @app.route('/MTH_RF', methods=['GET'])
# def MTH_RF():
#     acurracy, precision, recall, fscore, y_true, y_predict,cm = MTH.getRandomForest()
#     data = {
#         "accuracy": acurracy,
#         "precision": precision,
#         "recall": recall,
#         "fscore": fscore,
#         "y_true": y_true,
#         "y_predict": y_predict,
#         "cm": cm
#     }
#     return jsonify(data)

# @app.route('/MTH_STACK', methods=['GET'])
# def MTH_STACK():
#     acurracy, precision, recall, fscore, y_true, y_predict,cm = MTH.getStacking()
#     data = {
#         "accuracy": acurracy,
#         "precision": precision,
#         "recall": recall,
#         "fscore": fscore,
#         "y_true": y_true,
#         "y_predict": y_predict,
#         "cm": cm
#     }
#     return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)