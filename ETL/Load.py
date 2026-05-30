import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

try:

    df = pd.read_csv(r'C:\Users\indur\OneDrive\Desktop\power bi projects\Public_Transport_Delay\ETL\-Smart-Traffic-Accident-Risk-Prediction-System-Using-Machine-Learning\DATA\cleaned_smart_accident.csv')


    host = 'localhost'
    user = 'root'
    password = quote_plus('induri@05')
    database = 'public'

    create = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}')

    df.to_sql('smart',con=create,if_exists='replace',index=False)

except Exception as e:
    print(e)

finally:
    print("Successfully completed LOADED IN SQL WORKBENCH")                           
