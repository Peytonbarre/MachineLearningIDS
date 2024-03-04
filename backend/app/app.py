from flask import Flask, jsonify, request
from flask_cors import CORS
import MTH

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Welcome to IDS-ML!"

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