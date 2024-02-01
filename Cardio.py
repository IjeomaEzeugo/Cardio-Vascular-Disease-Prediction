import streamlit as st
import pandas as pd
import numpy as np
from pickle import dump
import pickle

model = pickle.load(open('cla_model.pkl','rb'))
Scaler = pickle.load(open('cla_scaler.pkl','rb'))



def data_table():
    C1, C2, C3,= st.columns(3)

    with C1:
        HT = st.number_input("What is your HEIGHT in cm?", step=100)
        WT = st.number_input('What is your WEIGHT in Kg', step =1)
        APHI = st.number_input('What is your Systolic blood pressure', step = 100)
        APLO = st.number_input('What is your Diastolic blood pressure',step=100)

    with C2:
        AGE = st.number_input('How old are you?', step=1)
        BMI = st.number_input('What is your Body Mass Index', step= 10)

        ACTIVE = st.selectbox("Are you involved in physical activites?",["YES","NO"])
        if ACTIVE=="Yes":
            ACTIVE1 = 1
        else:
            ACTIVE1 = 0

        ALCOHOL = st.selectbox("Do you take Alcohol?",["YES","NO"])
        if ALCOHOL=="Yes":
           ALCOHOL = 1
        else:
           ALCOHOL = 0

    with C3:    
        SMOKE = st.selectbox("Do you Smoke?",["YES","NO"])
        if SMOKE=="Yes":
           SMOKE = 1
        else:
           SMOKE = 0

        
        GENDER = st.selectbox("What is your GENDER?",["FEMALE","MALE"])
        if GENDER=="FEMALE":
            GENDER1 = 1
        else:
            GENDER1 = 0

       
        CHOL_LEVEL = st.selectbox("What is our Cholesterol level?",["NORMAL","ABOVE NORMAL","WELL ABOVE NORMAL"])
        if CHOL_LEVEL=="NORMAL":
            CHOL_LEVEL1 = 1
        else:
            CHOL_LEVEL1 = 2


        GLU_LEVEL = st.selectbox("What is your Glucose level?",["NORMAL","ABOVE NORMAL","WELL ABOVE NORMAL"])
        if GLU_LEVEL=="NORMAL":
            GLU_LEVEL1 = 1
        else:
            GLU_LEVEL1 = 2

 

    feat = np.array([HT,WT,APHI,APLO,AGE,GENDER1,CHOL_LEVEL1,ACTIVE1,ALCOHOL,SMOKE,GENDER1,GLU_LEVEL1,
                     ]).reshape(1,-1)
    
    cat =  ['age', 'gender', 'height', 'weight',
             'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 
             'smoke', 'alco', 'active', 'BMI']
    
    table = pd.DataFrame(feat,columns=cat)
    return table

def process(df):
     
    

    col = df.columns

    df = Scaler.fit_transform(df)

    df = pd.DataFrame(df,columns=col)
    
    

    return df


frame = data_table()

if st.button('Predict'):
    frame2 =process(frame)
    pred = model.predict(frame2)
    st.write(round(pred[0],0))