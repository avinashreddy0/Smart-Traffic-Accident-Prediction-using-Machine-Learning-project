import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


df.columns

df.drop(columns=['Unnamed: 0'],axis=True)

# FEATURES AND TARGET VALUES

x = df.drop('ACCIDENT_OCCURRENCE',axis=True)

y = df['ACCIDENT_OCCURRENCE']
df


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




#feature selection
print('logistic regression Coff_ ')

feature = models.named_steps['logistic']

importance = feature.coef_[0]
feature_names = preprocessing.get_feature_names_out()


importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': importance
})

importance_df = importance_df.sort_values(by = 'Coefficient',ascending=False)
print(importance_df)


plt.figure(figsize=(6,10))
sns.histplot(importance_df)
plt.title('feature selection:')
plt.tight_layout()
plt.show()


#feature selection



feature_names = random_model.named_steps['RFM']
importance = feature_names.feature_importances_

feature_name = preprocessing.get_feature_names_out()

importance_df = pd.DataFrame({
    'features':feature_name,
    'feature_secetion':importance
})
importance_df = importance_df.sort_values(by = 'feature_secetion',ascending=False)
print(importance_df)


plt.figure(figsize=(6,10))
sns.histplot(importance_df)
plt.title('feature selection random forest:')
plt.tight_layout()
plt.show()



feature = xg_model.named_steps['XGB']
importance = feature.feature_importances_

features_name = preprocessing.get_feature_names_out()

importance_df = pd.DataFrame({

    'feature':features_name,
    'importance':importance
})
importance_df = importance_df.sort_values(by ='importance',ascending=False)
print(importance_df)


plt.figure(figsize=(6,10))
sns.boxenplot(importance_df)
plt.title('feature section XGBoosting :')
plt.tight_layout()
plt.show()










x = preprocessing.fit_transform(x)
