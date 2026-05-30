import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'C:\Users\indur\OneDrive\Desktop\power bi projects\Public_Transport_Delay\ETL\-Smart-Traffic-Accident-Risk-Prediction-System-Using-Machine-Learning\DATA\cleaned_smart_accident.csv')
df


# STEP 1 BASIC  EDA TO UNDERSTAND THE DATA

print('statical summery')
print(df.describe().to_string())

print('information of data like dtypes,columns,null numbers')
print(df.info())

print('check dimensionality')
print(df.shape)

print('columns')
print(df.columns)




# STEP 2 CHECK DUPLICATES

print('Check Duplicates')
print(df.duplicated().sum())

print('Removing Duplicates')
print(df.drop_duplicates(df).to_string())


# STEP 3 CHECK MISSING VALUES

print('Checking messing values')
print(df.isnull().sum())


# STEP 4
print('Checking Imbalance')
imbalance = df['ACCIDENT_OCCURRENCE'].value_counts(normalize=True)*100
print(imbalance)


# STEP 5 UNDERSTANDING CATEGORICAL COLUMNS

categorical = df.groupby(['WEATHER_CONDITION','TRAFFIC_DENSITY','ROAD_CONDITION','ROAD_TYPE']).agg({

    'SPEED_LIMIT':'mean',
    'ACCIDENT_OCCURRENCE':'mean'
})

print(categorical.head().to_string())

# STEP 6 UNDERSTANDING NUMERICAL DATA


numerical = df.groupby('ROAD_TYPE')['ACCIDENT_OCCURRENCE'].mean()
print(numerical.head().to_string())

plt.figure(figsize=(10,6))
sns.barplot(numerical)
plt.title('compare the categorical values and numerical')
plt.xlabel('road_type')
plt.ylabel('ACCIDENT_OCCURRENCE')
plt.tight_layout()
plt.show()

#STEP 7 RELATIONSHIP PLOT
plt.Figure(figsize=(10,6))
sns.scatterplot(x='ACCIDENT_OCCURRENCE',y='SPEED_LIMIT',data=df)
plt.title('HOUR' 'vs' 'SPEED_LIMIT')
plt.xlabel('ACCIDENT_OCCURRENCE')
plt.ylabel('SPEED_LIMIT')
plt.tight_layout()

plt.show()



# STEP 8
print('Checking correlation between numerical features')
corr = df.corr(numeric_only=True)
print(corr.sample().to_string())

plt.Figure(figsize=(10,6))
sns.heatmap(corr)
plt.title('correlation between features')
plt.tight_layout()
plt.show()

# STEP 9 DISTRIBUTION VALUES

plt.Figure(figsize=(10,6))
sns.histplot(df['ACCIDENT_OCCURRENCE'])
plt.title('distribution of values')
plt.tight_layout()
plt.show()


# STEP 10 CHECKING OUTLIERS

plt.figure(figsize=(10,6))
sns.boxenplot(df['ACCIDENT_OCCURRENCE'])
plt.title('checking outliers of data')
plt.tight_layout()
plt.show()



# STEP 11 RELATION COMPLETE TABLE

plt.figure(figsize=(10,6))
sns.pairplot(df)
plt.tight_layout()
plt.show()



#STEP 12 COUNT VALUES

sns.countplot(df['ACCIDENT_OCCURRENCE'])
plt.show()