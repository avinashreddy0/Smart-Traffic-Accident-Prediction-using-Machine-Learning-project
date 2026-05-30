import pandas as pd
import matplotlib.pyplot as plt
import joblib
import streamlit as st
import plotly.express as px


st.set_page_config(page_icon='',page_title='smart traffic accident prediction',layout='centered')

st.title(':rainbow[Smart Traffic Accident prediction]')
#loading data ''

try:
    logistic = joblib.load('logistic_regression.pkl')
    random = joblib.load('random forest classification.pkl')
    xgboost = joblib.load('xgboost.pkl')
    preprocess = joblib.load('preprocessing.pkl')
except Exception as e:
    st.error('model')
    st.write(e)
    st.stop()

model_selection = st.sidebar.selectbox(r'select_model',['logistic regression','random forest classifier','XBBoost classifier'])

tab1,tab2,tab3,tab4,tab5 = st.tabs([
    'Welcome page',
    'About Project',
    'User Input',
    'Visualization',
    'About Developer'

])

with tab1:
  
    st.title('Welcome to :rainbow[ 🚦Smart Traffic Accident Prediction]')
    st.image('Gemini_Generated_Image_q86k8bq86k8bq86k.png')

with tab2:
   
    st.write(''' 🚦 Smart Traffic Accident Risk Prediction System

## Project Overview

This project predicts the probability of traffic accidents using Machine Learning models based on real-world traffic and environmental conditions.

The system analyzes multiple factors such as:

* Weather Condition
* Traffic Density
* Vehicle Speed
* Visibility
* Road Condition
* Road Type
* Speed Limit
* Time (Hour)
* Precipitation
* Traffic Signals
* Junction Information

Using these features, the model predicts whether the accident risk is:

* Low Risk
* High Risk

---

# 🔍 Project Workflow

## 1️⃣ ETL Pipeline

Collected, cleaned, and transformed real-world traffic data.

## 2️⃣ Data Preprocessing

Handled:

* Missing values
* Categorical encoding
* Feature scaling
* Data transformation

## 3️⃣ Feature Engineering

Created meaningful features such as:

* Night Driving Indicator
* Speed Risk Ratio
* Low Visibility Indicator
* Traffic Risk Score

## 4️⃣ Machine Learning Models

Implemented and compared:

* Logistic Regression
* Random Forest
* XGBoost

## 5️⃣ Model Evaluation

Evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix

## 6️⃣ Hyperparameter Tuning

Optimized models using:

* GridSearchCV
* Cross Validation

## 7️⃣ Explainable AI

Used SHAP Explainability to understand:

* feature importance
* prediction behavior
* accident risk contribution

---

# 📊 Dashboard & Visualization

Integrated:

* Power BI Dashboard
* Interactive Visualizations
* Risk Analysis Charts
* Feature Importance Analysis

---

# 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* XGBoost
* SHAP
* SQL & MySQL
* SQLAlchemy
* Power BI
* Streamlit

---

# 🎯 Goal of the Project

The main goal of this project is to build an intelligent traffic accident risk prediction system that can help:

* improve road safety
* identify high-risk conditions
* support traffic analysis
* assist data-driven decision making

---

# 🚀 Future Improvements

* Real-time prediction system
* Live traffic API integration
* Advanced dashboard analytics
* Cloud deployment
* GPS-based risk monitoring
             
''')
    
with tab3:
    with st.form('Click'):
        col1,col2 = st.columns(2)


        with col1:
            VEHICLE_SPEED = st.number_input('VEHICLE_SPEED',max_value=500)
            HOUR = st.number_input('HOUR',max_value=500)
            SPEED_LIMIT = st.number_input('SPEED_LIMIT',max_value=500)
            VISIBILITY	 = st.number_input('VISIBILITY',max_value=100.0)
            JUNCTION = st.selectbox('JUNCTION',[1,0])
            TRAFFIC_SIGNAL = st.selectbox('TRAFFIC_SIGNAL',[0,1])
        with col2:
            WEATHER_CONDITION = st.selectbox('WEATHER_CONDITION',['Foggy','Snow','Rainy','Misty','Storm','Sunny','Clearsky'])
            TRAFFIC_DENSITY = st.selectbox('TRAFFIC_DENSITY',['Traffichigh','Low','Jampacked','Medtraffic','Hgh','Light','Meddensity','Lowtraffic'])
            ROAD_CONDITION = st.selectbox('ROAD_CONDITION',['Roughroad','Waterwater','Dryroad','Icy','Muddy'])
            ROAD_TYPE = st.selectbox('ROAD_TYPE',['City','Bridgelane','Urbanroad','Rural','Expressway','Serviceroad'])
            PRECIPITATION = st.selectbox('PRECIPITATION',['Lightrain','Norain','Fogdrizzle','Heavyrain','Snowfall'])

        submit = st.form_submit_button('predict Accident Risk')

    if submit:
        try:
            HIGH_TRAFFIC =  1 if TRAFFIC_DENSITY in [
                'Traffichigh',
                'Jampacked',
                'High'
            ] else 0

            LOW_VISIBILITY = 1 if VISIBILITY < 10 else 0

            IS_NIGHT = 1 if HOUR >= 18 or HOUR <= 5 else 0


            input_df = pd.DataFrame([{

                'Unnamed: 0':0,
                'WEATHER_CONDITION':WEATHER_CONDITION,
                'VISIBILITY':VISIBILITY,
                'TRAFFIC_DENSITY':TRAFFIC_DENSITY,
                'VEHICLE_SPEED':VEHICLE_SPEED,
                'ROAD_CONDITION':ROAD_CONDITION,
                'HOUR':HOUR,
                'SPEED_LIMIT':SPEED_LIMIT,
                'ROAD_TYPE':ROAD_TYPE,
                'PRECIPITATION':PRECIPITATION,
                'JUNCTION':JUNCTION,
                'TRAFFIC_SIGNAL':TRAFFIC_SIGNAL,
                'LOW_VISIBILITY':LOW_VISIBILITY,
                'IS_NIGHT':IS_NIGHT,
                'HIGH_TRAFFIC':HIGH_TRAFFIC
            }])

            input_processed = preprocess.transform(input_df)
            

            if model_selection == 'logistic regression':
                y_pred = logistic.predict(input_processed)[0]
                y_pred_prob = logistic.predict_proba(input_processed)[0]
                risk_score = float(y_pred_prob[1])
            elif model_selection == 'random forest classifier':
                y_pred = random.predict(input_processed)[0]
                y_pred_prob = random.predict_proba(input_processed)[0]
                risk_score = float(y_pred_prob[1])
            elif model_selection == 'XBBoost classifier':
                y_pred = xgboost.predict(input_processed)[0]
                y_pred_prob = xgboost.predict_proba(input_processed)[0]
                risk_score = float(y_pred_prob[1])


            if y_pred == 1:
                st.progress(int(risk_score* 100))
                st.write(f'Risk Score:{risk_score:.2f}')
                st.error('High Accident Risk')
                st.image('OIPA.jpg',width=300)
                st.snow()

            else:
                st.progress(int(risk_score*100))
                st.write(f'Risk Score:{risk_score:.2f}')
                st.success('Low Accident Risk')
                st.image('OIP (1).jpg')
                st.snow()
        except Exception as e:
            st.error('error During prediction')
            st.write(e)
            

        

        
with tab4:

   
    st.title('Smart Traffic Accident Prediction VIsualization')

    data = pd.read_csv(r'C:\Users\indur\OneDrive\Desktop\power bi projects\Public_Transport_Delay\ETL\-Smart-Traffic-Accident-Risk-Prediction-System-Using-Machine-Learning\DATA\feature_engineering.csv')
    st.dataframe(data.head(10))

    st.metric('total Records',len(data))

    st.write("IS_NIGHT 0 ans 1")
    IS_NIGHT = data['IS_NIGHT'].value_counts(normalize=True)
    st.bar_chart(IS_NIGHT)

    st.write('WEATHER_CONDITION')

    fig = px.histogram(data,x='WEATHER_CONDITION',color='ACCIDENT_OCCURRENCE')
    st.plotly_chart(fig)

    st.write('HOUR')

    fig = px.line(data,x= 'HOUR')
    st.plotly_chart(fig)


  

    cal = data['VEHICLE_SPEED'].mean()
    st.metric('Average Vehicle Speed',round(cal,2))


    st.subheader('RISK SCORE DISTRIBUTION')

    cal =  data['RISK_SCORE'].value_counts()
    st.bar_chart(cal)
        
       

    


with tab5:
   
    st.title(":rainbow[ABOUT DEVELOPER]")

    st.markdown('''# 👨‍💻 About Developer

Hi, I’m Induri Avinash Reddy, a passionate aspiring Data Scientist and AIML undergraduate student focused on building real-world Machine Learning and Data Analytics projects.

I am currently working on end-to-end AI and Data Science projects involving:

* Machine Learning
* Data Analysis
* ETL Pipelines
* SQL & MySQL
* Power BI Dashboards
* Explainable AI (SHAP)
* Streamlit Applications

This Smart Traffic Accident Risk Prediction System was developed to analyze traffic and environmental conditions and predict accident risk using Machine Learning models such as Logistic Regression, Random Forest, and XGBoost.

The project combines:

* data preprocessing
* feature engineering
* model evaluation
* hyperparameter tuning
* explainable AI
* dashboard visualization
* deployment workflows

My goal is to build practical AI solutions that are simple, data-driven, and impactful.

## Skills & Technologies

Python | Pandas | NumPy | Scikit-learn | XGBoost | SQL | MySQL | Power BI | Streamlit | SHAP | ETL


## Interests

* Data Science
* Machine Learning
* AI Applications
* Analytics
* Real-World Problem Solvingcd 
* Dashboard Development

🚀 Continuously learning and building projects to strengthen practical industry-level Data Science skills.

''')
    

    
    st.markdown(

    "[portfolio](https://avinashreddy0.github.io/portfolio-website/)"
)
 
    st.markdown(  "[Email](mailto:induriavinashreddy05@gmail.com)"
)
    st.markdown(
    "[GitHub](https://github.com/avinashreddy0/indurireddy)")
    st.markdown(
    "[LinkedIn](https://www.linkedin.com/in/avinash-reddy-induri-4662b832a)")
    st.markdown(
    "[Project Repository](https://github.com/avinashreddy0/Public_Transport_Delay)"

)

    st.markdown('''
   
    Name : Avinash Reddy Induri
            

    Email:induriavinashreddy05@gmail.com
            

    Phone number : 9346739650
            

    location: Guntur
            

    B Tech Final Year -> AIML
            

    Aspiring  Data Scientist  & Machine learning  & Data Analysis        
    

''')


st.caption('❤️ Streamlit Deploy')

    





