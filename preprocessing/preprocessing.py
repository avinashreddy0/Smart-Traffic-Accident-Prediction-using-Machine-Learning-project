import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split,cross_val_predict,RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score,precision_score,confusion_matrix
import shap
import joblib

import warnings
warnings.filterwarnings('ignore')


# LOADING DATA

df = pd.read_csv(r'C:\Users\indur\OneDrive\Desktop\power bi projects\Public_Transport_Delay\ETL\-Smart-Traffic-Accident-Risk-Prediction-System-Using-Machine-Learning\DATA\cleaned_smart_accident.csv')
df
# FEATURES AND TARGET VALUES

x = df.drop('ACCIDENT_OCCURRENCE',axis=True)

y = df['ACCIDENT_OCCURRENCE']

# num and categorical

num = x.select_dtypes(include=['int64','float64']).columns
cat = [['WEATHER_CONDITION','TRAFFIC_DENSITY','ROAD_CONDITION','ROAD_TYPE','PRECIPITATION']]


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

print(preprocessing)