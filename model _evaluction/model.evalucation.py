import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split,cross_val_predict,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score,precision_score,accuracy_score,f1_score,confusion_matrix,roc_auc_score
import shap
import joblib

import warnings
warnings.filterwarnings('ignore')


# LOADING DATA

df = pd.read_csv(r'C:\Users\indur\OneDrive\Desktop\power bi projects\Public_Transport_Delay\ETL\-Smart-Traffic-Accident-Risk-Prediction-System-Using-Machine-Learning\DATA\feature_engineering.csv')
df


# FEATURES AND TARGET VALUES

x = df.drop('ACCIDENT_OCCURRENCE',axis=True)

y = df['ACCIDENT_OCCURRENCE']
df

df.columns

df.drop(columns=['Unnamed: 0'],axis=True)


y = df['ACCIDENT_OCCURRENCE']
print(y.shape)
print(type(y))


# num and categorical

num = x.select_dtypes(include=['int64']).columns
cat = x.select_dtypes(include='object').columns


#pipe line for numerical columns

num_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())
])


# pipe line fro categorical columns 

cat_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('one_hot',OneHotEncoder(handle_unknown='ignore'))
])

#   preprocessing using COLUMN TRANSFORM

preprocessing = ColumnTransformer([

    ('num_col',num_pipeline,num),
    ('cat_col',cat_pipeline,cat)
])

x = preprocessing.fit_transform(x)


# splitting the data 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#  MODEL PIPELINES LOGISTIC REGRESSION

models = Pipeline([
    ('logistic',LogisticRegression(random_state=42)),
    
])

random_model = Pipeline([
    ('RFM',RandomForestClassifier())
])

xg_model = Pipeline([
    ('XGB',XGBClassifier())
])


# liner model training

models.fit(x_train,y_train)
y_pred_lin = models.predict(x_test)
print(f'logistic regression:{y_pred_lin}')
y_prob_lin = models.predict_proba(x_test)[:,1]
print(f'predict_prob:{y_prob_lin}')

# random forest training
random_model.fit(x_train,y_train)
y_pred=random_model.predict(x_test)
print(f'random forest classification:{y_pred}')
y_prob_rf = models.predict_proba(x_test)[:,1]
print(f'predict_prob:{y_prob_rf}')


#XGBoost model training

xg_model.fit(x_train,y_train)
XG_y_pred=xg_model.predict(x_train)
print(f'XGBoost :{XG_y_pred}')
y_prob_xg = models.predict_proba(x_test)[:,1]
print(f'predict_prob:{y_prob_xg}')



# evaluation metrics
# logistic regression


print(f'LogisticRegression evaluation metics:')

acc_score = accuracy_score(y_test,y_pred_lin)
pr_score = precision_score(y_test,y_pred_lin)
con_matrix =confusion_matrix(y_test,y_pred_lin)
roc_aucscore = roc_auc_score(y_test,y_prob_lin)
f1score = f1_score(y_test,y_pred_lin)

print(f'accuracy_score:{acc_score}')

print(f'precision_score:{pr_score}')
print(f'confusion_matrix:{con_matrix}')
print(f'roc_auc_score:{roc_aucscore}')
print(f'f1_score:',f1_score)


y_thershold = (y_prob_lin > 0.3).astype(int)
re_score = recall_score(y_test,y_thershold)
print(f'recall_score:{re_score}')


# random forest classification

print(f'random forest evaluation metrics:')

print('accuracy_score:',accuracy_score(y_test,y_pred))
print('precision_score:',precision_score(y_test,y_pred))
print('f1_score:',f1_score(y_test,y_pred))
print('confus_matrix:',confusion_matrix(y_test,y_pred))
print('roc_auc_score:',roc_auc_score(y_test,y_prob_rf))



#handling thershold

y_the = (y_prob_rf > 0.3).astype(int)

print(f'recall_score:',recall_score(y_test,y_the))



# xgboost evaluation metics:

print('XGBClassifier evaluation metics:')

print('roc_auc_score:',roc_auc_score(y_test,y_prob_xg))
print('confus_matrix:',confusion_matrix(y_test,y_pred))
print('precision_score:',precision_score(y_test,y_pred))
print('f1_score:',f1_score(y_test,y_pred))
print('accuracy_score:',accuracy_score(y_test,y_pred))

#thershold

y_the = (y_prob_xg > 0.3).astype(int)

print('recall_score:',recall_score(y_test,y_the))
      