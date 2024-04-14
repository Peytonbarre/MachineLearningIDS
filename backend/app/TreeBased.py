import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import plot_importance
from imblearn.over_sampling import SMOTE
from sqlalchemy import create_engine #pip install sqlalchemy
#pip install mysqlclient

using_stacking = False
y_test_stacking = []
y_train_stacking = []

modified_datafile = True
df = pd.DataFrame()

engine = create_engine('mysql://admin:projectt60@csproject.c5emwcgweqq7.us-east-2.rds.amazonaws.com/data')
query = "SELECT * FROM csvdata"

#After selecing features, train and split again
def dataframeSetup():
    global modified_datafile

    if not modified_datafile:
        #Read dataset
        # csv_paths = ["./backend/app/data/CICIDS2017_sample_km.csv"]
        # csv_selected = list(np.zeros(len(csv_paths)))

        # Using main csv file
        # csv_selected[0] = 1

        # Select data file that user decided on
        # df = pd.read_csv(csv_paths[csv_selected.index(1)])
        global df
        df = pd.read_sql(query, engine)
        
        # Randomly sample instances from majority classes
        df_minor = df[(df['Label']=='WebAttack')|(df['Label']=='Bot')|(df['Label']=='Infiltration')]
        df_BENIGN = df[(df['Label']=='BENIGN')]
        df_BENIGN = df_BENIGN.sample(n=None, frac=0.01, replace=False, weights=None, random_state=None, axis=0)
        df_DoS = df[(df['Label']=='DoS')]
        df_DoS = df_DoS.sample(n=None, frac=0.05, replace=False, weights=None, random_state=None, axis=0)
        df_PortScan = df[(df['Label']=='PortScan')]
        df_PortScan = df_PortScan.sample(n=None, frac=0.05, replace=False, weights=None, random_state=None, axis=0)
        df_BruteForce = df[(df['Label']=='BruteForce')]
        df_BruteForce = df_BruteForce.sample(n=None, frac=0.2, replace=False, weights=None, random_state=None, axis=0)

        df_s = df_BENIGN._append(df_DoS)._append(df_PortScan)._append(df_BruteForce)._append(df_minor)
        df_s = df_s.sort_index()

        print("===")
        print(df_BENIGN)

        # Save the sampled dataset
        df_s.to_csv('./backend/app/data/CICIDS2017_sample.csv',index=0)
        modified_datafile = False
    
    df = pd.read_csv('./backend/app/data/CICIDS2017_sample.csv')

    # Min-max normalization
    numeric_features = df.dtypes[df.dtypes != 'object'].index
    df[numeric_features] = df[numeric_features].apply(
        lambda x: (x - x.min()) / (x.max()-x.min()))
    # Fill empty values by 0
    df = df.fillna(0)

    labelencoder = LabelEncoder()
    df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])

def applyDefaultHyperparameters(train_size, smote_sampling_strategy):
    global df
    dataframeSetup()


    print(df)
    X = df.drop(['Label'],axis=1).values 
    y = df.iloc[:, -1].values.reshape(-1,1)
    y=np.ravel(y)
    trainSizePlaceholder = float(train_size)
    testSizePlaceholder = round((1 - trainSizePlaceholder), 2)
    print(trainSizePlaceholder)
    print(testSizePlaceholder)
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = trainSizePlaceholder, test_size = testSizePlaceholder, random_state = 0,stratify = y)

    pairs = smote_sampling_strategy.split(",")
    sampling_dict = {}
    
    for pair in pairs:
        key, value = pair.split(":")
        sampling_dict[int(key)] = int(value)

    smote=SMOTE(n_jobs=-1,sampling_strategy=sampling_dict) # Default: create 1500 samples for the minority class "4"

    X_train, y_train = smote.fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test

def GET_TB_DECISION_TREE(train_size = 0.8, smote_sampling_strategy = "4:1500", random_state = 0):
    X_train, X_test, y_train, y_test = applyDefaultHyperparameters(train_size, smote_sampling_strategy)
    # Decision tree training and prediction
    dt = DecisionTreeClassifier(random_state = random_state)
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
    # cm=confusion_matrix(y_true,y_predict)
    # f,ax=plt.subplots(figsize=(5,5))
    # sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    # plt.xlabel("y_pred")
    # plt.ylabel("y_true")
    # plt.show()

    dt_train=dt.predict(X_train)
    dt_test=dt.predict(X_test)
    return dt_train, dt_test, dt_score, precision, recall, fscore

def GET_TB_RANDOM_FOREST(train_size = 0.8, smote_sampling_strategy = "4:1500", random_state = 0):
    X_train, X_test, y_train, y_test = applyDefaultHyperparameters(train_size, smote_sampling_strategy)
    # Random Forest training and prediction
    rf = RandomForestClassifier(random_state = random_state)
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
    # cm=confusion_matrix(y_true,y_predict)
    # f,ax=plt.subplots(figsize=(5,5))
    # sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    # plt.xlabel("y_pred")
    # plt.ylabel("y_true")
    # plt.show()

    rf_train=rf.predict(X_train)
    rf_test=rf.predict(X_test)
    return rf_train, rf_test, rf_score, precision, recall, fscore

def GET_TB_EXTRA_TREES(train_size = 0.8, smote_sampling_strategy = "4:1500", random_state = 0):
    X_train, X_test, y_train, y_test = applyDefaultHyperparameters(train_size, smote_sampling_strategy)
    # Extra trees training and prediction
    et = ExtraTreesClassifier(random_state = random_state)
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
    # cm=confusion_matrix(y_true,y_predict)
    # f,ax=plt.subplots(figsize=(5,5))
    # sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    # plt.xlabel("y_pred")
    # plt.ylabel("y_true")
    # plt.show()

    et_train=et.predict(X_train)
    et_test=et.predict(X_test)
    return et_train, et_test, et_score, precision, recall, fscore

def GET_TB_XGBOOST(train_size = 0.8, smote_sampling_strategy = "4:1500", n_estimators = 10):
    X_train, X_test, y_train, y_test = applyDefaultHyperparameters(train_size, smote_sampling_strategy)
    # XGboost training and prediction
    xg = xgb.XGBClassifier(n_estimators = n_estimators)
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
    # cm=confusion_matrix(y_true,y_predict)
    # f,ax=plt.subplots(figsize=(5,5))
    # sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    # plt.xlabel("y_pred")
    # plt.ylabel("y_true")
    # plt.show()

    xg_train=xg.predict(X_train)
    xg_test=xg.predict(X_test)
    return xg_train, xg_test, xg_score, precision, recall, fscore

def GET_TB_STACKING(train_size = 0.8, smote_sampling_strategy = "4:1500"):
    global using_stacking, y_test_stacking, y_train_stacking
    using_stacking = True
    y_test_stacking = []
    y_train_stacking = []
    
    dt_train, dt_test, dt_score, dt_precision, dt_recall, dt_fscore = GET_TB_DECISION_TREE(train_size, smote_sampling_strategy)
    et_train, et_test, et_score, et_precision, et_recall, et_fscore = GET_TB_EXTRA_TREES(train_size, smote_sampling_strategy)
    rf_train, rf_test, rf_score, rf_precision, rf_recall, rf_fscore = GET_TB_RANDOM_FOREST(train_size, smote_sampling_strategy)
    xg_train, xg_test, xg_score, xg_precision, xg_recall, xg_fscore = GET_TB_XGBOOST(train_size, smote_sampling_strategy)
    # Use the outputs of 4 base models to construct a new ensemble model
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
    # f,ax=plt.subplots(figsize=(5,5))
    # sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    # plt.xlabel("y_pred")
    # plt.ylabel("y_true")
    # plt.show()

    #Return for TreeBased stacking
    #Line graph
    AvgofEvent = [
        #dt
        [dt_score, dt_precision, dt_recall, dt_fscore],
        #rf
        [rf_score, rf_precision, rf_recall, rf_fscore],
        #et
        [et_score, et_precision, et_recall, et_fscore],
        #xg
        [xg_score, xg_precision, xg_recall, xg_fscore],
        #stk
        [stk_score, precision, recall, fscore],
    ]

    #Bar Graph
    precisionScores = [dt_precision, rf_precision, et_precision, xg_precision, precision]
    f1Scores = [dt_fscore, rf_fscore, et_fscore, xg_fscore, fscore]
    recallScores = [dt_recall, rf_recall, et_recall, xg_recall, recall]
    accuracyScores = [dt_score, rf_score, et_score, xg_score, stk_score]

    #Pie Chart
    return(stk_score, precision.tolist(), recall.tolist(), fscore.tolist(), cm.tolist(), AvgofEvent, precisionScores, f1Scores, recallScores, accuracyScores)

def FEATURE_SELECTION(train_size = 0.8, smote_sampling_strategy = "4:1500"): # Note: NOT A MODEL ALGORITHM
    global df
    # Save the feature importance lists generated by four tree-based algorithms
    dt_feature = dt.feature_importances_
    rf_feature = rf.feature_importances_
    et_feature = et.feature_importances_
    xgb_feature = xg.feature_importances_

    # calculate the average importance value of each feature
    avg_feature = (dt_feature + rf_feature + et_feature + xgb_feature)/4

    dataframeSetup()
    feature=(df.drop(['Label'],axis=1)).columns.values
    print ("Features sorted by their score:")
    print (sorted(zip(map(lambda x: round(x, 4), avg_feature), feature), reverse=True))

    f_list = sorted(zip(map(lambda x: round(x, 4), avg_feature), feature), reverse=True)

    # Select the important features from top-importance to bottom-importance until the accumulated importance reaches 0.9 (out of 1)
    Sum = 0
    fs = []
    for i in range(0, len(f_list)):
        Sum = Sum + f_list[i][0]
        fs.append(f_list[i][1])
        if Sum>=0.9:
            break      

    # X_fs = df[fs].values
    # X_train, X_test, y_train, y_test = train_test_split(X_fs, y, train_size = 0.8, test_size = 0.2, random_state = 0,stratify = y)

    # from imblearn.over_sampling import SMOTE
    # smote=SMOTE(n_jobs=-1,sampling_strategy={4:1500})

    # X_train, y_train = smote.fit_resample(X_train, y_train)
    X_train, X_test, y_train, y_test = applyDefaultHyperparameters(train_size, smote_sampling_strategy)

    pd.Series(y_train).value_counts()

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
    # cm=confusion_matrix(y_true,y_predict)
    # f,ax=plt.subplots(figsize=(5,5))
    # sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    # plt.xlabel("y_pred")
    # plt.ylabel("y_true")
    # plt.show()

    dt_train=dt.predict(X_train)
    dt_test=dt.predict(X_test)

    rf = RandomForestClassifier(random_state = 0)
    rf.fit(X_train,y_train) # modelin veri üzerinde öğrenmesi fit fonksiyonuyla yapılıyor
    rf_score=rf.score(X_test,y_test)
    y_predict=rf.predict(X_test)
    y_true=y_test
    print('Accuracy of RF: '+ str(rf_score))
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    print('Precision of RF: '+(str(precision)))
    print('Recall of RF: '+(str(recall)))
    print('F1-score of RF: '+(str(fscore)))
    print(classification_report(y_true,y_predict))
    # cm=confusion_matrix(y_true,y_predict)
    # f,ax=plt.subplots(figsize=(5,5))
    # sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    # plt.xlabel("y_pred")
    # plt.ylabel("y_true")
    # plt.show()

    rf_train=rf.predict(X_train)
    rf_test=rf.predict(X_test)

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
    # cm=confusion_matrix(y_true,y_predict)
    # f,ax=plt.subplots(figsize=(5,5))
    # sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    # plt.xlabel("y_pred")
    # plt.ylabel("y_true")
    # plt.show()

    et_train=et.predict(X_train)
    et_test=et.predict(X_test)

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
    # cm=confusion_matrix(y_true,y_predict)
    # f,ax=plt.subplots(figsize=(5,5))
    # sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    # plt.xlabel("y_pred")
    # plt.ylabel("y_true")
    # plt.show()

    xg_train=xg.predict(X_train)
    xg_test=xg.predict(X_test)

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
    print('Accuracy of Stacking: '+ str(stk_score))
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    print('Precision of Stacking: '+(str(precision)))
    print('Recall of Stacking: '+(str(recall)))
    print('F1-score of Stacking: '+(str(fscore)))
    print(classification_report(y_true,y_predict))
    # cm=confusion_matrix(y_true,y_predict)
    # f,ax=plt.subplots(figsize=(5,5))
    # sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    # plt.xlabel("y_pred")
    # plt.ylabel("y_true")
    # plt.show()
    
    return f_list

GET_TB_STACKING()