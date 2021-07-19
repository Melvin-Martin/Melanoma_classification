import tensorflow as tf
model = tf.keras.models.load_model('model.h5')
import streamlit as st
import pandas as pd
import base64
main_bg_ext = "png"
main_bg = "bg.png"
hide_streamlit_style = ""
st.markdown(
		f"""
		<style>
		.reportview-container {{
 		background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
		background-size: cover;
		}}
		</style>
		""", unsafe_allow_html=True) 
st.markdown(hide_streamlit_style, unsafe_allow_html=True);
col1, col2, col3 = st.beta_columns([1,6,1])

with col1:
    st.write("")

with col2:
    st.write("""
         # Melanoma Classification
         """
         )

with col3:
    st.write("")


st.write("Upload skin lession images for predictions")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

import cv2
from PIL import Image, ImageOps
import numpy as np

def import_and_predict(image_data, model):
    image = Image.open(file)
    
    image = np.asarray(image)
    img_resize = (cv2.resize(image, dsize=(450, 600)))
    
    img_resize = img_resize/255.0
    img_reshape = img_resize[np.newaxis,...]
        
    prediction = model.predict(img_reshape)
    
    return prediction

if file is None:
    st.text("No images added")
else:
    
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        txt = str(int(prediction[0,0]*100))+'% '+"Actinic keratosis" 
        st.write(txt)
    elif np.argmax(prediction) == 1:
        txt = str(int(prediction[0,1]*100))+'% '+'Basal cell carcinoma'
        st.write(txt)
    elif np.argmax(prediction) == 2:
        txt = str(int(prediction[0,2]*100))+'% '+"Benign keratosis"
        st.write(txt)
    elif np.argmax(prediction) == 3:
        txt = str(int(prediction[0,3]*100))+'% '+"Dermatofibroma"
        st.write(txt)
    elif np.argmax(prediction) == 4:
        txt = str(int(prediction[0,4]*100))+'% '+"Melanoma"
        st.write(txt)
    elif np.argmax(prediction) == 5:
        txt = str(int(prediction[0,5]*100))+'% '+"Melanocytic nevus"
        st.write(txt)
    elif np.argmax(prediction) == 6:
        txt = str(int(prediction[0,6]*100))+'% '+"Squamous cell carcinoma"
        st.write(txt)
    elif np.argmax(prediction) == 7:
        txt = str(int(prediction[0,7]*100))+'% '+"Vascular lesion"
        st.write(txt)

    pred =  pd.DataFrame(prediction)
    pred.columns = ["Actinic keratosis","Basal cell carcinoma","Benign keratosis","Dermatofibroma",
                    "Melanoma","Melanocytic nevus","Squamous cell carcinoma","Vascular lesion"]
    st.write(pred,use_column_width=True)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    