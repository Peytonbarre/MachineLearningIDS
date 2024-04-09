import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
from xgboost import plot_importance
from FCBF_module import FCBF, FCBFK, FCBFiP, get_i
from imblearn.over_sampling import SMOTE
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sqlalchemy import create_engine #pip install sqlalchemy
#pip install mysqlclient

using_stacking = False
y_test_stacking = []
y_train_stacking = []
engine = create_engine('mysql://admin:projectt60@csproject.c5emwcgweqq7.us-east-2.rds.amazonaws.com/data')
query = "SELECT * FROM csvdata"

#After selecing features, train and split again
def applyDefaultHyperparameters(train_size, smote_sampling_strategy):
    global using_stacking, y_test_stacking, y_train_stacking

    #Reading sample dataset
    #df=pd.read_csv('./backend/app/data/CICIDS2017_sample_km.csv')
    df = pd.read_sql(query, engine)

    #Dropping labels and reshaping
    X = df.drop(['Label'],axis=1).values
    y = df.iloc[:, -1].values.reshape(-1,1)
    y=np.ravel(y)

    #Splitting test and train into 20 80 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 0, stratify = y)

    if using_stacking:
        y_test_stacking.append(y_test)
        y_train_stacking.append(y_train)

    #calculating importance scores
    importances = mutual_info_classif(X_train, y_train)
    f_list = sorted(zip(map(lambda x: round(x, 4), importances), features), reverse=True)
    Sum = 0
    fs = []
    for i in range(0, len(f_list)):
        Sum = Sum + f_list[i][0]
        fs.append(f_list[i][1])
    X_fs = df[fs].values
    X_fs.shape

    #selecting features using the FCBF algorithm
    fcbf = FCBFK(k = 20)
    X_fss = fcbf.fit_transform(X_fs,y)
    X_fss.shape
    
    X_train, X_test, y_train, y_test = train_test_split(X_fss,y, train_size = float(train_size)/100, test_size = (1 - float(train_size)/100), random_state = 0,stratify = y)
    X_train.shape
    pd.Series(y_train).value_counts()

    pairs = smote_sampling_strategy.split(",")
    sampling_dict = {}
    
    for pair in pairs:
        key, value = pair.split(":")
        sampling_dict[int(key)] = int(value)

    smote=SMOTE(n_jobs=-1,sampling_strategy=sampling_dict)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    pd.Series(y_train).value_counts()
    return X_train, X_test, y_train, y_test

def XGBoost(train_size = 0.8, smote_sampling_strategy = "2:1000, 4:1000"):
    X_train, X_test, y_train, y_test = applyDefaultHyperparameters(train_size, smote_sampling_strategy)
    xg = xgb.XGBClassifier(n_estimators = 10)
    xg.fit(X_train,y_train)
    xg_score=xg.score(X_test,y_test)
    y_predict=xg.predict(X_test)
    y_true=y_test
    print('Accuracy of XGBoost: '+ str(xg_score))
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    print('Precision of XGBoost: '+(str(precision)))
    print('Recall of XGBoost: '+(str(recall)))
    print('F1-score of XGBoost: '+(str(fscore)))
    print(classification_report(y_true,y_predict))
    cm=confusion_matrix(y_true,y_predict)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)

def HPO_BO_TPE_XGBOOST(train_size = 0.8, smote_sampling_strategy = "2:1000, 4:1000"):
    X_train, X_test, y_train, y_test = applyDefaultHyperparameters(train_size, smote_sampling_strategy)

    def objective(params):
        params = {
            'n_estimators': int(params['n_estimators']), 
            'max_depth': int(params['max_depth']),
            'learning_rate':  abs(float(params['learning_rate'])),

        }
        clf = xgb.XGBClassifier( **params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = accuracy_score(y_test, y_pred)

        return {'loss':-score, 'status': STATUS_OK }

    space = {
        'n_estimators': hp.quniform('n_estimators', 10, 100, 5),
        'max_depth': hp.quniform('max_depth', 4, 100, 1),
        'learning_rate': hp.normal('learning_rate', 0.01, 0.9),
    }

    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=20)
    print("XGBoost: Hyperopt estimated optimum {}".format(best))

    xg = xgb.XGBClassifier(learning_rate= 0.7340229699980686, n_estimators = 70, max_depth = 14)
    xg.fit(X_train,y_train)
    xg_score=xg.score(X_test,y_test)
    y_predict=xg.predict(X_test)
    y_true=y_test
    print('Accuracy of XGBoost: '+ str(xg_score))
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    print('Precision of XGBoost: '+(str(precision)))
    print('Recall of XGBoost: '+(str(recall)))
    print('F1-score of XGBoost: '+(str(fscore)))
    print(classification_report(y_true,y_predict))
    cm=confusion_matrix(y_true,y_predict)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)


    rf = RandomForestClassifier(random_state = 0)
    rf.fit(X_train,y_train) 
    rf_score=rf.score(X_test,y_test)
    y_predict=rf.predict(X_test)
    y_true=y_test
    print('Accuracy of RF: '+ str(rf_score))
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    print('Precision of RF: '+(str(precision)))
    print('Recall of RF: '+(str(recall)))
    print('F1-score of RF: '+(str(fscore)))
    print(classification_report(y_true,y_predict))
    cm=confusion_matrix(y_true,y_predict)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    xg_train=xg.predict(X_train)
    xg_test=xg.predict(X_test)
    return xg_train, xg_test

def HPO_BO_TPE_FOREST(train_size = 0.8, smote_sampling_strategy = "2:1000, 4:1000"):
    X_train, X_test, y_train, y_test = applyDefaultHyperparameters(train_size, smote_sampling_strategy)

    def objective2(params):
        params = {
            'n_estimators': int(params['n_estimators']), 
            'max_depth': int(params['max_depth']),
            'max_features': int(params['max_features']),
            "min_samples_split":int(params['min_samples_split']),
            "min_samples_leaf":int(params['min_samples_leaf']),
            "criterion":str(params['criterion'])
        }
        clf = RandomForestClassifier( **params)
        clf.fit(X_train,y_train)
        score=clf.score(X_test,y_test)

        return {'loss':-score, 'status': STATUS_OK }
    # Define the hyperparameter configuration space
    space = {
        'n_estimators': hp.quniform('n_estimators', 10, 200, 1),
        'max_depth': hp.quniform('max_depth', 5, 50, 1),
        "max_features":hp.quniform('max_features', 1, 20, 1),
        "min_samples_split":hp.quniform('min_samples_split',2,11,1),
        "min_samples_leaf":hp.quniform('min_samples_leaf',1,11,1),
        "criterion":hp.choice('criterion',['gini','entropy'])
    }

    best = fmin(fn=objective2,
                space=space,
                algo=tpe.suggest,
                max_evals=20)
    print("Random Forest: Hyperopt estimated optimum {}".format(best))

    rf_hpo = RandomForestClassifier(n_estimators = 71, min_samples_leaf = 1, max_depth = 46, min_samples_split = 9, max_features = 20, criterion = 'entropy')
    rf_hpo.fit(X_train,y_train)
    rf_score=rf_hpo.score(X_test,y_test)
    y_predict=rf_hpo.predict(X_test)
    y_true=y_test
    print('Accuracy of RF: '+ str(rf_score))
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    print('Precision of RF: '+(str(precision)))
    print('Recall of RF: '+(str(recall)))
    print('F1-score of RF: '+(str(fscore)))
    print(classification_report(y_true,y_predict))
    cm=confusion_matrix(y_true,y_predict)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    rf_train=rf_hpo.predict(X_train)
    rf_test=rf_hpo.predict(X_test)

    dt = DecisionTreeClassifier(random_state = 0)
    dt.fit(X_train,y_train) 
    dt_score=dt.score(X_test,y_test)
    y_predict=dt.predict(X_test)
    y_true=y_test
    print('Accuracy of DT: '+ str(dt_score))
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    print('Precision of DT: '+(str(precision)))
    print('Recall of DT: '+(str(recall)))
    print('F1-score of DT: '+(str(fscore)))
    print(classification_report(y_true,y_predict))
    cm=confusion_matrix(y_true,y_predict)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    return rf_train, rf_test

def HPO_BO_TPE_DECISION_TREE(train_size = 0.8, smote_sampling_strategy = "2:1000, 4:1000"):
    X_train, X_test, y_train, y_test = applyDefaultHyperparameters(train_size, smote_sampling_strategy)

    def objective(params):
        params = {
            'max_depth': int(params['max_depth']),
            'max_features': int(params['max_features']),
            "min_samples_split":int(params['min_samples_split']),
            "min_samples_leaf":int(params['min_samples_leaf']),
            "criterion":str(params['criterion'])
        }
        clf = DecisionTreeClassifier( **params)
        clf.fit(X_train,y_train)
        score=clf.score(X_test,y_test)

        return {'loss':-score, 'status': STATUS_OK }
    # Define the hyperparameter configuration space
    space = {
        'max_depth': hp.quniform('max_depth', 5, 50, 1),
        "max_features":hp.quniform('max_features', 1, 20, 1),
        "min_samples_split":hp.quniform('min_samples_split',2,11,1),
        "min_samples_leaf":hp.quniform('min_samples_leaf',1,11,1),
        "criterion":hp.choice('criterion',['gini','entropy'])
    }

    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=50)
    print("Decision tree: Hyperopt estimated optimum {}".format(best))
    dt_hpo = DecisionTreeClassifier(min_samples_leaf = 2, max_depth = 47, min_samples_split = 3, max_features = 19, criterion = 'gini')
    dt_hpo.fit(X_train,y_train)
    dt_score=dt_hpo.score(X_test,y_test)
    y_predict=dt_hpo.predict(X_test)
    y_true=y_test
    print('Accuracy of DT: '+ str(dt_score))
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    print('Precision of DT: '+(str(precision)))
    print('Recall of DT: '+(str(recall)))
    print('F1-score of DT: '+(str(fscore)))
    print(classification_report(y_true,y_predict))
    cm=confusion_matrix(y_true,y_predict)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    dt_train=dt_hpo.predict(X_train)
    dt_test=dt_hpo.predict(X_test)
    et = ExtraTreesClassifier(random_state = 0)
    et.fit(X_train,y_train) 
    et_score=et.score(X_test,y_test)
    y_predict=et.predict(X_test)
    y_true=y_test
    print('Accuracy of ET: '+ str(et_score))
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    print('Precision of ET: '+(str(precision)))
    print('Recall of ET: '+(str(recall)))
    print('F1-score of ET: '+(str(fscore)))
    print(classification_report(y_true,y_predict))
    cm=confusion_matrix(y_true,y_predict)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    return dt_train, dt_test

def HPO_BO_TPE_EXTRA_TREES(train_size = 0.8, smote_sampling_strategy = "2:1000, 4:1000"):
    X_train, X_test, y_train, y_test = applyDefaultHyperparameters(train_size, smote_sampling_strategy)

    def objective(params):
        params = {
            'n_estimators': int(params['n_estimators']), 
            'max_depth': int(params['max_depth']),
            'max_features': int(params['max_features']),
            "min_samples_split":int(params['min_samples_split']),
            "min_samples_leaf":int(params['min_samples_leaf']),
            "criterion":str(params['criterion'])
        }
        clf = ExtraTreesClassifier( **params)
        clf.fit(X_train,y_train)
        score=clf.score(X_test,y_test)

        return {'loss':-score, 'status': STATUS_OK }
    # Define the hyperparameter configuration space
    space = {
        'n_estimators': hp.quniform('n_estimators', 10, 200, 1),
        'max_depth': hp.quniform('max_depth', 5, 50, 1),
        "max_features":hp.quniform('max_features', 1, 20, 1),
        "min_samples_split":hp.quniform('min_samples_split',2,11,1),
        "min_samples_leaf":hp.quniform('min_samples_leaf',1,11,1),
        "criterion":hp.choice('criterion',['gini','entropy'])
    }

    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=20)
    print("Random Forest: Hyperopt estimated optimum {}".format(best))
    et_hpo = ExtraTreesClassifier(n_estimators = 53, min_samples_leaf = 1, max_depth = 31, min_samples_split = 5, max_features = 20, criterion = 'entropy')
    et_hpo.fit(X_train,y_train) 
    et_score=et_hpo.score(X_test,y_test)
    y_predict=et_hpo.predict(X_test)
    y_true=y_test
    print('Accuracy of ET: '+ str(et_score))
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    print('Precision of ET: '+(str(precision)))
    print('Recall of ET: '+(str(recall)))
    print('F1-score of ET: '+(str(fscore)))
    print(classification_report(y_true,y_predict))
    cm=confusion_matrix(y_true,y_predict)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    et_train=et_hpo.predict(X_train)
    et_test=et_hpo.predict(X_test)
    return et_train, et_test

def stacking(train_size = 0.8, smote_sampling_strategy = "2:1000, 4:1000"):
    global using_stacking, y_test_stacking, y_train_stacking
    using_stacking = True
    y_test_stacking = []
    y_train_stacking = []
    
    dt_train, dt_test = HPO_BO_TPE_DECISION_TREE(train_size, smote_sampling_strategy)
    et_train, et_test = HPO_BO_TPE_EXTRA_TREES(train_size, smote_sampling_strategy)
    rf_train, rf_test = HPO_BO_TPE_FOREST(train_size, smote_sampling_strategy)
    xg_train, xg_test = HPO_BO_TPE_XGBOOST(train_size, smote_sampling_strategy)
    base_predictions_train = pd.DataFrame( {
    'DecisionTree': dt_train.ravel(),
        'RandomForest': rf_train.ravel(),
     'ExtraTrees': et_train.ravel(),
     'XgBoost': xg_train.ravel(),
    })
    base_predictions_train.head(5)
    dt_train=dt_train.reshape(-1, 1)
    et_train=et_train.reshape(-1, 1)
    rf_train=rf_train.reshape(-1, 1)
    xg_train=xg_train.reshape(-1, 1)
    dt_test=dt_test.reshape(-1, 1)
    et_test=et_test.reshape(-1, 1)
    rf_test=rf_test.reshape(-1, 1)
    xg_test=xg_test.reshape(-1, 1)
    x_train = np.concatenate(( dt_train, et_train, rf_train, xg_train), axis=1)
    x_test = np.concatenate(( dt_test, et_test, rf_test, xg_test), axis=1)
    stk = xgb.XGBClassifier().fit(x_train, y_train_stacking)
    y_predict=stk.predict(x_test)
    y_true=y_test_stacking
    stk_score=accuracy_score(y_true,y_predict)
    print('Accuracy of Stacking: '+ str(stk_score))
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    print('Precision of Stacking: '+(str(precision)))
    print('Recall of Stacking: '+(str(recall)))
    print('F1-score of Stacking: '+(str(fscore)))
    print(classification_report(y_true,y_predict))
    cm=confusion_matrix(y_true,y_predict)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)

def getXGBoost(train_size = 0.8, smote_sampling_strategy = "2:1000, 4:1000"):
    X_train, X_test, y_train, y_test = applyDefaultHyperparameters(train_size, smote_sampling_strategy)
    xg = xgb.XGBClassifier(n_estimators = 10)
    xg.fit(X_train,y_train)
    xg_score=xg.score(X_test,y_test)
    y_predict=xg.predict(X_test)
    y_true=y_test
    acurracy = xg_score
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    cm=confusion_matrix(y_true,y_predict)
    # f,ax=plt.subplots(figsize=(5,5))
    # sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    return(acurracy, precision.tolist(), recall.tolist(), fscore.tolist(), y_true.tolist(), y_predict.tolist(), cm.tolist())

def getExtraTrees(train_size = 0.8, smote_sampling_strategy = "2:1000, 4:1000"):
    X_train, X_test, y_train, y_test = applyDefaultHyperparameters(train_size, smote_sampling_strategy)

    def objective(params):
        params = {
            'n_estimators': int(params['n_estimators']), 
            'max_depth': int(params['max_depth']),
            'max_features': int(params['max_features']),
            "min_samples_split":int(params['min_samples_split']),
            "min_samples_leaf":int(params['min_samples_leaf']),
            "criterion":str(params['criterion'])
        }
        clf = ExtraTreesClassifier( **params)
        clf.fit(X_train,y_train)
        score=clf.score(X_test,y_test)

        return {'loss':-score, 'status': STATUS_OK }
    # Define the hyperparameter configuration space
    space = {
        'n_estimators': hp.quniform('n_estimators', 10, 200, 1),
        'max_depth': hp.quniform('max_depth', 5, 50, 1),
        "max_features":hp.quniform('max_features', 1, 20, 1),
        "min_samples_split":hp.quniform('min_samples_split',2,11,1),
        "min_samples_leaf":hp.quniform('min_samples_leaf',1,11,1),
        "criterion":hp.choice('criterion',['gini','entropy'])
    }
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=20)
    et_hpo = ExtraTreesClassifier(n_estimators = 53, min_samples_leaf = 1, max_depth = 31, min_samples_split = 5, max_features = 20, criterion = 'entropy')
    et_hpo.fit(X_train,y_train) 
    et_score=et_hpo.score(X_test,y_test)
    acurracy = et_score
    y_predict=et_hpo.predict(X_test)
    y_true=y_test
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    cm=confusion_matrix(y_true,y_predict)
    # f,ax=plt.subplots(figsize=(5,5))
    # sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    return(acurracy, precision.tolist(), recall.tolist(), fscore.tolist(), y_true.tolist(), y_predict.tolist(), cm.tolist())

def getDecisionTree(train_size = 0.8, smote_sampling_strategy = "2:1000, 4:1000"):
    X_train, X_test, y_train, y_test = applyDefaultHyperparameters(train_size, smote_sampling_strategy)

    def objective(params):
        params = {
            'max_depth': int(params['max_depth']),
            'max_features': int(params['max_features']),
            "min_samples_split":int(params['min_samples_split']),
            "min_samples_leaf":int(params['min_samples_leaf']),
            "criterion":str(params['criterion'])
        }
        clf = DecisionTreeClassifier( **params)
        clf.fit(X_train,y_train)
        score=clf.score(X_test,y_test)

        return {'loss':-score, 'status': STATUS_OK }
    # Define the hyperparameter configuration space
    space = {
        'max_depth': hp.quniform('max_depth', 5, 50, 1),
        "max_features":hp.quniform('max_features', 1, 20, 1),
        "min_samples_split":hp.quniform('min_samples_split',2,11,1),
        "min_samples_leaf":hp.quniform('min_samples_leaf',1,11,1),
        "criterion":hp.choice('criterion',['gini','entropy'])
    }

    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=50)
    dt_hpo = DecisionTreeClassifier(min_samples_leaf = 2, max_depth = 47, min_samples_split = 3, max_features = 19, criterion = 'gini')
    dt_hpo.fit(X_train,y_train)
    dt_score=dt_hpo.score(X_test,y_test)
    y_predict=dt_hpo.predict(X_test)
    y_true=y_test
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    cm=confusion_matrix(y_true,y_predict)
    # f,ax=plt.subplots(figsize=(5,5))
    # sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    dt_train=dt_hpo.predict(X_train)
    dt_test=dt_hpo.predict(X_test)
    et = ExtraTreesClassifier(random_state = 0)
    et.fit(X_train,y_train) 
    et_score=et.score(X_test,y_test)
    accuracy = et_score
    y_predict=et.predict(X_test)
    y_true=y_test
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    cm=confusion_matrix(y_true,y_predict)
    # f,ax=plt.subplots(figsize=(5,5))
    # sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    return(accuracy, precision.tolist(), recall.tolist(), fscore.tolist(), y_true.tolist(), y_predict.tolist(), cm.tolist())

def getRandomForest(train_size = 0.8, smote_sampling_strategy = "2:1000, 4:1000"):
    X_train, X_test, y_train, y_test = applyDefaultHyperparameters(train_size, smote_sampling_strategy)

    def objective2(params):
        params = {
            'n_estimators': int(params['n_estimators']), 
            'max_depth': int(params['max_depth']),
            'max_features': int(params['max_features']),
            "min_samples_split":int(params['min_samples_split']),
            "min_samples_leaf":int(params['min_samples_leaf']),
            "criterion":str(params['criterion'])
        }
        clf = RandomForestClassifier( **params)
        clf.fit(X_train,y_train)
        score=clf.score(X_test,y_test)

        return {'loss':-score, 'status': STATUS_OK }
    space = {
        'n_estimators': hp.quniform('n_estimators', 10, 200, 1),
        'max_depth': hp.quniform('max_depth', 5, 50, 1),
        "max_features":hp.quniform('max_features', 1, 20, 1),
        "min_samples_split":hp.quniform('min_samples_split',2,11,1),
        "min_samples_leaf":hp.quniform('min_samples_leaf',1,11,1),
        "criterion":hp.choice('criterion',['gini','entropy'])
    }
    best = fmin(fn=objective2,
                space=space,
                algo=tpe.suggest,
                max_evals=20)
    rf_hpo = RandomForestClassifier(n_estimators = 71, min_samples_leaf = 1, max_depth = 46, min_samples_split = 9, max_features = 20, criterion = 'entropy')
    rf_hpo.fit(X_train,y_train)
    rf_score=rf_hpo.score(X_test,y_test)
    accuracy = rf_score
    y_predict=rf_hpo.predict(X_test)
    y_true=y_test
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    cm=confusion_matrix(y_true,y_predict)
    # f,ax=plt.subplots(figsize=(5,5))
    # sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    rf_train=rf_hpo.predict(X_train)
    rf_test=rf_hpo.predict(X_test)
    dt = DecisionTreeClassifier(random_state = 0)
    dt.fit(X_train,y_train) 
    dt_score=dt.score(X_test,y_test)
    y_predict=dt.predict(X_test)
    y_true=y_test
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    cm=confusion_matrix(y_true,y_predict)
    # f,ax=plt.subplots(figsize=(5,5))
    # sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    return(accuracy, precision.tolist(), recall.tolist(), fscore.tolist(), y_true.tolist(), y_predict.tolist(), cm.tolist())

def getStacking(train_size = 0.8, smote_sampling_strategy = "2:1000, 4:1000"):
    X_train, X_test, y_train, y_test = applyDefaultHyperparameters(train_size, smote_sampling_strategy)

    dt_train, dt_test = HPO_BO_TPE_DECISION_TREE(train_size, smote_sampling_strategy)
    print("Decision Tree Complete")
    rf_train, rf_test = HPO_BO_TPE_FOREST(train_size, smote_sampling_strategy)
    print("Random Forest Complete")
    et_train, et_test = HPO_BO_TPE_EXTRA_TREES(train_size, smote_sampling_strategy)
    print("ET Complete")
    xg_train, xg_test = HPO_BO_TPE_XGBOOST(train_size, smote_sampling_strategy)
    print("XGBoost Complete")
    base_predictions_train = pd.DataFrame( {
    'DecisionTree': dt_train.ravel(),
        'RandomForest': rf_train.ravel(),
     'ExtraTrees': et_train.ravel(),
     'XgBoost': xg_train.ravel(),
    })
    base_predictions_train.head(5)
    dt_train=dt_train.reshape(-1, 1)
    et_train=et_train.reshape(-1, 1)
    rf_train=rf_train.reshape(-1, 1)
    xg_train=xg_train.reshape(-1, 1)
    dt_test=dt_test.reshape(-1, 1)
    et_test=et_test.reshape(-1, 1)
    rf_test=rf_test.reshape(-1, 1)
    xg_test=xg_test.reshape(-1, 1)
    x_train = np.concatenate(( dt_train, et_train, rf_train, xg_train), axis=1)
    x_test = np.concatenate(( dt_test, et_test, rf_test, xg_test), axis=1)
    stk = xgb.XGBClassifier().fit(x_train, y_train)
    y_predict=stk.predict(x_test)
    y_true=y_test
    stk_score=accuracy_score(y_true,y_predict)
    accuracy = stk_score
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    cm=confusion_matrix(y_true,y_predict)
    # f,ax=plt.subplots(figsize=(5,5))
    # sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    return(accuracy, precision.tolist(), recall.tolist(), fscore.tolist(), y_true.tolist(), y_predict.tolist(), cm.tolist())