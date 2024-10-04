import streamlit as st
import pandas as pd
import datetime 
import pickle
cars_df = pd.read_csv('./car_pred.csv')
st.write(
    """
    # Cars24 Used Cars Prediction
    
    """
)

st.dataframe(cars_df.head())

fuel_type= st.selectbox(
    "Select fuel_type",
    ("Petrol", "Diesel", "Electric","CNG"),
)

st.write("You selected:",fuel_type)

engine = st.slider("Set the Engine power", 500, 5000,step= 100)

Transmission_type= st.selectbox(
    "Select Transmission Type",
    ("Manual", "Automatic")
)

Seats= st.selectbox(
    "No of seats",
    (2,4,5,6,7,8))

# input features = [[2018.0,1,4000,Fuel_type,Transmission_type,19.70,emgine,86.30,seats]]
encode_dict = {
    "fuel_type":{"Diesel":1,"Petrol":2,"CNG":3,"LPG":4,"Electric":5},
    "Seller_type":{"Dealer":1,"Individual":2,"Trust mark Dealer":3},
    "Transmission_type" :{"Automatic":1,"Manual":2},
    
}

def model_pred(fuel_type,transmission_type,engine,seats):
    with open("car_pred",'rb') as file:
              reg_model=pickle.load(file)
    input_features = [[2018.0,1,4000,fuel_type, Transmission_type,
                       19.70,engine,86.30,seats]]
    return reg_model.predict(input_features)

if (st.button("Predict Price")):
    fuel_type = encode_dict["fuel_type"][fuel_type]
    Transmission_type = encode_dict["Transmission_type"][Transmission_type]
    Price = model_pred(fuel_type,Transmission_type,engine,Seats)
    st.text(f"Price of the car is {Price[0]} Lakh Rupees")