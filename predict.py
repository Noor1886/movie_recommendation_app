import streamlit as st 
import pickle
import numpy as np 

def load_model():
    with open('saved_steps.pkl','rb') as file:
        data = pickle.load(file)
    return data 

data = load_model()

regressor = data['model']



def show_predict_page():
    st.title(" Movie Recommendation Algorithm")
    
    
    st.write(" we need more info")