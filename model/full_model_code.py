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

# random forest cross_valuation

RB_cross_valudation = cross_val_predict(random_model,x,y,cv=5)
print(RB_cross_valudation)



# XGBOOST cross_valudataion

XGB_cross_valudatation = cross_val_predict(xg_model,x,y,cv=5)
print(XGB_cross_valudatation)

# liner model training

models.fit(x_train,y_train)
y_pred_lin = models.predict(x_test)
print(f'logistic regression:{y_pred_lin}')
y_prob_lin = models.predict_proba(x_test)[:,1]
print(f'predict_prob:{y_prob_lin}')


feature_name = preprocessing.get_feature_names_out()

result = pd.DataFrame(
    x_test.toarray(),
    columns=feature_name
)

result['Actual'] = y_test.values
result['prediction'] = y_pred_lin
print(result.to_string())



# random forest training
random_model.fit(x_train,y_train)
y_pred=random_model.predict(x_test)
print(f'random forest classification:{y_pred}')
y_prob_rf = models.predict_proba(x_test)[:,1]
print(f'predict_prob:{y_prob_rf}')

feature_name = preprocessing.get_feature_names_out()

result = pd.DataFrame(
    x_test.toarray(),
    columns=feature_name
)
result['Actual'] = y_test.values
result['prediction'] = y_pred
print(result.to_string())


#XGBoost model training

xg_model.fit(x_train,y_train)
XG_y_pred=xg_model.predict(x_train)
print(f'XGBoost :{XG_y_pred}')
y_prob_xg = models.predict_proba(x_test)[:,1]
print(f'predict_prob:{y_prob_xg}')


feature_name = preprocessing.get_feature_names_out()
XG_y_pred = xg_model.predict(x_test)

xg_result = pd.DataFrame(
    x_test.toarray(),
    columns=feature_name
)

xg_result['Actual'] = y_test.values

xg_result['prediction'] = XG_y_pred
print(xg_result.to_string())

combine_result = pd.DataFrame({

    'Actual':y_test.values,
    'LogisticRegression':y_pred_lin,
    'RandomForestClassifier':y_pred,
    'XGBClassifier':XG_y_pred
})

print(combine_result.to_string())



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
      


# Hyper parameters 

print('Hyper parameters:logistic regression')
grid_para = {

    'logistic__C':[0.01,0.1,1,10],
    'logistic__max_iter':[500,1000],
    'logistic__penalty':['l2']
}
grid_search_cv  = GridSearchCV(models,param_grid=grid_para,cv = 5)
grid_search_cv.fit(x_train,y_train)
print(grid_search_cv.best_params_)
print(grid_search_cv.best_estimator_)



#hyper parameters

print('Hyper Parameters:Random Forest ')

para_grid = {

    'RFM__n_estimators':[100,200],
    'RFM__max_depth':[3,7,8],
    'RFM__min_samples_split':[2,5,6]
}
ride_search_cv = GridSearchCV(random_model,para_grid,cv=5,scoring='recall')
ride_search_cv.fit(x_train,y_train)
print(ride_search_cv.best_params_)
print(ride_search_cv.best_estimator_)


#hyper parameters

print('Hyper Parameters:Random Forest ')

para_grid = {

    'RFM__n_estimators':[100,200],
    'RFM__max_depth':[3,7,8],
    'RFM__min_samples_split':[2,5,6]
}
ride_search_cv = GridSearchCV(random_model,para_grid,cv=5,scoring='recall')
ride_search_cv.fit(x_train,y_train)
print(ride_search_cv.best_params_)
print(ride_search_cv.best_estimator_)




#SHAP

xgb_model = xg_model.named_steps['XGB']
feature_names = preprocessing.get_feature_names_out()
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(x_train)
summary_plot = shap.summary_plot(shap_values,x_train.toarray(),feature_names=feature_names)



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
sns.boxenplot(importance_df)
plt.title('feature section logisticregresssion :')
plt.tight_layout()
plt.show()



#feature selection

print('feature selection RandomForestClassifier')

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
plt.title('feature selection:random forest Classifier')
plt.tight_layout()
plt.show()

#feature selection

print('feature selection RandomForestClassifier')

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
plt.title('feature selection:random forest Classifier')
plt.tight_layout()
plt.show()

logistic_model = joblib.dump(models,'logistic_regression.pkl')
random_forest = joblib.dump(random_model,'random forest classification.pkl')
xgboost = joblib.dump(xg_model,'xgboost.pkl')


joblib.dump(preprocessing,'preprocessing.pkl')






























joblib.dump(preprocessing,'preprocessing.pkl')


