import pandas as pd

try:
    data = pd.read_csv(r'C:\Users\indur\OneDrive\Desktop\power bi projects\Public_Transport_Delay\ETL\-Smart-Traffic-Accident-Risk-Prediction-System-Using-Machine-Learning\DATA\messy_smart_traffic_accident_dataset.csv')
    print(data.to_string())
except FileNotFoundError:
    print('PLEASE CHECK ONE FILE PATH????????????????????????????')
finally:
    print('WE SUCCESSFULLY LOADED THE ')