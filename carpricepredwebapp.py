# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 22:04:52 2023

@author: nikitha
"""

import numpy as np
import pickle 
import streamlit as st



loaded_model=pickle.load(open("car_prediction_model.sav",'rb'))

#creating a function for classification
def car_price_prediction(input_data):
    
    numpy_array= np.asarray(input_data,dtype=object)
    
    reshaped = numpy_array.reshape(1,-1)
    
    pred = loaded_model.predict(reshaped)
    
    pred_round = '%.2f' % pred[0]
    
    return f'{pred_round} Lakhs'
   
        
def main():
    #giving a title
    st.title('Used Car Price Prediction Web App')
   
    #getting the input from user
    Year=st.text_input('Enter the car model year')
    
    Present_Price=st.text_input('Enter the present price of the car')
    
    Kilometers=st.text_input('Enter the Kilometers Driven')
    
    Fuel_type=st.text_input('Enter the fuel type: 0 for CNG, 1 for Diesel, 2 for Petrol')
    
    Seller_type=st.text_input('Enter the seller type: 0 for Dealer, 1 for Individual')
    
    Transmission=st.text_input('Enter the transmission: 0 for Automactic, 1 for Manual')
    
    Owner=st.text_input('Enter the number of previous owners: 0/1/3')
    
    #code for prediction 
    Prediction=''
    
    #creating a button for prediction
    if st.button('Car Price Prediction Result'):
        Prediction=car_price_prediction([[Year,Present_Price,Kilometers,Fuel_type,Seller_type,Transmission,Owner]])
        
    st.success(Prediction)
    
if __name__ =='__main__':
    main()
    