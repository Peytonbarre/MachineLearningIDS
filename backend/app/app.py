import os
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from sqlalchemy import create_engine, text
from datetime import datetime
import pandas as pd
import json
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
    trainValue = round(float(data.get('slideValue'))/100,2)
    classifier = data.get('classifier')
    graphType = data.get('graphType')
    parameter = data.get('parameter')
  
    #TODO When done with metric derivation, make this SQL statement add ALL metrics for each query (move below)
    # Use parameterized query to prevent SQL injection
    query = text("SELECT * FROM history WHERE Model = :classifier AND smote = :SMOTE AND trainVal = CAST(:trainValue as FLOAT)")
    result = pd.read_sql(query, engine, params={"classifier": classifier, "SMOTE": SMOTE, "trainValue": trainValue}) 

    if not result.empty:
        print("Already found! " + str(result.iloc[0]))
        print(result.iloc[0]['TimeStamps'])
        data = [result.iloc[0]['Accuracy'], result.iloc[0]['Precisions'], result.iloc[0]['Recall'], result.iloc[0]['F1_Score'], result.iloc[0]['CM'], result.iloc[0]['Avg_of_event'], result.iloc[0]['PrecisionScores'], result.iloc[0]['f1Scores'], result.iloc[0]['recallScores'], result.iloc[0]['accuracyScores']]
    else:
        print("Not found!")
        if classifier == 'MTH':
            data = MTH.getStacking(trainValue, SMOTE)
        #TODO: Make sure this is working
        elif classifier == 'LCCDE':
            data = LCCDE.applyDefaultHyperparameters(trainValue, SMOTE)
        #TODO: Make sure this is working
        elif classifier == 'Tree-Based':
            data = TreeBased.applyDefaultHyperparameters(trainValue, SMOTE)

        currentTime = datetime.now()
        addQuery = text("INSERT INTO history (TimeStamps, Model, Accuracy, Precisions, Recall, F1_Score, CM, smote, trainVal, Avg_of_event, PrecisionScores, f1Scores, recallScores, accuracyScores) VALUES (:currentTime, :classifier, :Accuracy, :Precisions, :Recall, :F1_Score, :CM, :SMOTE, :trainValue, :Avg_of_event, :PrecisionScores, :f1Scores, :recallScores, :accuracyScores)")
        with engine.connect() as conn:
            conn.execute(addQuery, {"currentTime": currentTime, "classifier": classifier, "Accuracy": data[0], "Precisions": data[1], "Recall": data[2], "F1_Score": data[3], "CM": json.dumps(data[4]), "SMOTE": SMOTE, "trainValue": trainValue, "Avg_of_event": json.dumps(data[5]), "PrecisionScores": json.dumps(data[6]), "f1Scores": json.dumps(data[7]) , "recallScores": json.dumps(data[8]), "accuracyScores": json.dumps(data[9])})
            conn.commit()
    
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
        data = data[4]
    elif graphType == 'Line':
        data = data[5]
    elif graphType == 'Bar':
        if parameter == 'Precision By Classifier':
            data = data[6]
        elif parameter == 'Recall By Classifier':
            data = data[8]
        elif parameter == 'F-1 Score By Classifier':
            data = data[7]
        elif parameter == 'Accuracy By Classifier':
            data = data[9]
    elif graphType == 'Pie':
        #TODO: Change to classifier composition
        data = data[6]
    elif graphType == 'Callout':
        if parameter == 'Avg Accuracy':
            data = data[0]
        elif parameter == 'Avg Precision':
            data = data[1]
        elif parameter == 'Avg Recall':
            data = data[2]
        elif parameter == 'Avg F1':
            data = data[3]

    print("RETURNING: " + str(data))
    print("FOR: " + graphType)

    response = {
        "status": "good",
        "data": data
    }

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