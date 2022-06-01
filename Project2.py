import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import pickle

def Pro2():
    st.write("This a ML model for predicting car price using Random Forest regressor with Hyperparameter Tuning on Car Dekho dataset")

    df = pd.read_csv(r'./data/car data.csv')
    final_df = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
                   'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
    # Creating new column for driving year old cars
    final_df['No_year_old'] = 2022 - final_df['Year']

    st.title('Car Price Prediction')

    with st.sidebar:
        st.write("Select your choice.")

    Year = st.sidebar.number_input(
        'Select Year',
        min_value=2005, max_value=2022
    )
    Present_Price = st.sidebar.number_input(
        'Enter Present Value',
        min_value=0.5, max_value=final_df.Present_Price.max()
    )

    Kms_Driven = st.sidebar.number_input(
        'Enter Kms Driven',
        min_value=1000, max_value=50000
    )
    Kms_Driven2 = np.log(Kms_Driven)
    Owner = st.sidebar.selectbox(
        'Select No. of Owners',
        (final_df["Owner"].unique())
    )
    Fuel_Type = st.sidebar.selectbox(
        'Select Fuel Type',
        (["Petrol", "Diesel", "CNG"])
    )
    Fuel_Type_Petrol = 0
    Fuel_Type_Diesel = 0

    if (Fuel_Type == 'Petrol'):
        Fuel_Type_Petrol = 1
        Fuel_Type_Diesel = 0
    elif(Fuel_Type == 'Diesel'):
        Fuel_Type_Petrol = 0
        Fuel_Type_Diesel = 1
    elif (Fuel_Type == 'CNG'):
        Fuel_Type_Petrol = 0
        Fuel_Type_Diesel = 0

    Year = 2022 - Year

    Seller_Type_Individual = st.sidebar.selectbox(
        'Are you A Dealer or Individual',
        (["Individual", "Dealer"])
    )

    if (Seller_Type_Individual == 'Individual'):
        Seller_Type_Individual = 1
    else:
        Seller_Type_Individual = 0

    Transmission_Mannual = st.sidebar.selectbox(
        'Transmission type',
        (["Manual Car", "Automatic Car"])
    )
    if (Transmission_Mannual == 'Mannual'):
        Transmission_Mannual = 1
    else:
        Transmission_Mannual = 0


    st.subheader(
        f"You have selected a Car {Year} years old with present price of {Present_Price} lakhs, {Fuel_Type} version.")
    model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))
    prediction = model.predict([[Present_Price, Kms_Driven2, Owner, Year, Fuel_Type_Diesel, Fuel_Type_Petrol,
                                 Seller_Type_Individual, Transmission_Mannual]])
    output = round(prediction[0], 2)
    if output <= 0:
        st.write("Sorry you cannot sell this car")
    else:
        st.write(f"You Can Sell The Car in {output} lakhs.")



    def load_data1(nrows):
        data = pd.read_csv(r'car data.csv', nrows=nrows)
        lowercase = lambda x: str(x).lower()
        data.rename(lowercase, axis='columns', inplace=True)
        return data



    if st.checkbox('Show raw data', key="P2_1"):
        st.subheader('Raw data')
        data = load_data1(1000)
        st.dataframe(data)
        st.write('Shape of dataset:', df.shape)

    if st.checkbox('Show me EDA', key="P2_2"):
        st.text("Simple EDA of raw data")
        col1, col2, col3 = st.columns(3)

        with col1:
            bar_df = df.Fuel_Type.value_counts()
            st.bar_chart(bar_df, width=10, height=200)
        with col2:
            bar_df = df.Transmission.value_counts()
            st.bar_chart(bar_df, width=10, height=200)
        with col3:
            bar_df = df.Seller_Type.value_counts()
            st.bar_chart(bar_df, width=10, height=200)

