import pandas as pd

# Load the dataset
df = pd.read_csv('data.csv')
print(df.head())
#Dropping columns that are not needed
df = df.drop(columns=['id', 'Unnamed: 32'])

#Map the target to binary values: 'M' to 1 (malignant), 'B' to 0 (benign)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Separate features and target datasets
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=102)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

params = {
    "solver": "lbfgs",
    "max_iter": 10000,
    "multi_class": "auto",
    "random_state": 8888,
}

#train the model
model = LogisticRegression(**params)
model.fit(X_train, y_train)

#Predict and evaluate the model
y_pred = model.predict(X_test)
print("Logistic Regression:")
print(classification_report(y_test, y_pred))

class_report = classification_report(y_test, y_pred,output_dict= True)

#model with random forest
from sklearn.ensemble import RandomForestClassifier
rf_params = {
    'n_estimators': 200,           
    'criterion': 'entropy',      
    'max_depth': 15,              
    'min_samples_split': 5,       
    'min_samples_leaf': 3,         
    'max_features': 'sqrt',       
    'bootstrap': True,            
    'random_state': 42            
}

rf_model = RandomForestClassifier(**rf_params)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
print("Random forest")
print(classification_report(y_test, y_pred))

class_report_rf = classification_report(y_test, y_pred,output_dict= True)


# with xgboost
# xgboost_params = {
#     'objective': 'multi:softprob',
#     'eval_metric': 'mlogloss',
#     'num_class': 3,
#     'max_depth': 6,
#     'learning_rate': 0.1,
#     'n_estimators': 100,
#     'subsample': 0.8,
#     'colsample_bytree': 0.8,
#     'random_state': 42
# }
# import xgboost as xgb
# xgb_model = xgb.XGBClassifier(**xgboost_params)
# xgb_model.fit(X_train, y_train)

# xgb_y_pred = xgb_model.predict(X_test)
# class_report_xg = classification_report(y_test, xgb_y_pred, output_dict=True)

# Define metrics
# metrics = {
#     'accuracy': class_report_xg['accuracy'],
#     'recall_class_0': class_report_xg['0']['recall'],
#     'recall_class_1': class_report_xg['1']['recall'],
#     'recall_class_2': class_report_xg['2']['recall'],
#     'f1_score': class_report_xg['macro avg']['f1-score']
# }
# # 

import mlflow

mlflow.set_experiment("cancer_data")
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")

with mlflow.start_run(run_name="logistic"):
    mlflow.log_params(params)
    mlflow.log_metrics({
        'accuracy': class_report['accuracy'],
        'recall_class_0': class_report['0']['recall'],
        'recall_class_1': class_report['1']['recall'],
        'f1_score': class_report['macro avg']['f1-score']
        })
    mlflow.sklearn.log_model(model, "Logistic Regression")
    

with mlflow.start_run(run_name="RandomForestRun"):
    mlflow.log_params(rf_params)
    mlflow.log_metrics({
        'accuracy': class_report_rf['accuracy'],
        'recall_class_0': class_report_rf['0']['recall'],
        'recall_class_1': class_report_rf['1']['recall'],
        'f1_score': class_report_rf['macro avg']['f1-score']
        })
    mlflow.sklearn.log_model(rf_model, "Random forest")

# with mlflow.start_run(run_name="xgboost-experiment"):
#     mlflow.log_params(xgboost_params)
#     mlflow.log_metrics({
#         'accuracy': class_report_xg['accuracy'],
#         'recall_class_0': class_report_xg['0']['recall'],
#         'recall_class_1': class_report_xg['1']['recall'],
#         'f1_score': class_report_xg['macro avg']['f1-score']
#         })
#     mlflow.sklearn.log_model(xgb_model, "xgboost")    