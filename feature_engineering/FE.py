import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\indur\OneDrive\Desktop\power bi projects\Public_Transport_Delay\ETL\-Smart-Traffic-Accident-Risk-Prediction-System-Using-Machine-Learning\DATA\cleaned_smart_accident.csv')
df

#feature engineering

# 1.group by 

risk = df.groupby(['WEATHER_CONDITION','TRAFFIC_DENSITY','PRECIPITATION'])['ACCIDENT_OCCURRENCE'].mean().reset_index()
risk.rename(
    columns = {
        'ACCIDENT_OCCURRENCE':'RISK_SCORE'
    },
    inplace=True
)

df = df.merge(risk,on= ['WEATHER_CONDITION','TRAFFIC_DENSITY','PRECIPITATION'],).round(2)

df


#RADIO

df['SPEED_RISK'] = (df['SPEED_LIMIT'] / df['VEHICLE_SPEED']).round(2)
df

df['LOW_VISIBILITY'] = (df['VISIBILITY'] < 3).astype(int)


df['IS_NIGHT'] =(df['HOUR'] >= 20) | (df['HOUR'] <= 5).astype(int)
df['IS_NIGHT'] = df['IS_NIGHT'].astype(int)

HIGH_TRAFFIC =[

    'High',
    'Medtraffic',
    'Jampacked',
    'Low'
    'Meddensity'
    'Lowtraffic	'

]

df['HIGH_TRAFFIC'] = df['TRAFFIC_DENSITY'].isin(HIGH_TRAFFIC).astype(int)

df['HOUR'] =df['HOUR'].astype(str).str[:2]
df['HOUR'] =df['HOUR'].astype(int)

df.replace([np.inf,-np.inf],np.nan,inplace=True)

df.fillna(df.median(numeric_only=True),inplace=True)



df.to_csv('feature_engineering.csv')
df.to_excel('Smart_traffic_accident_prediction_using_ML.xlsx')