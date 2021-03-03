import streamlit as st
import tensorflow as tf

st.set_option("deprecation.showfileUploaderEncoding",False) # whenever installing streamlit app it shows warning to avoid it we put it false

# when we load our big model and if there exist some changes it reloads it which is time consuming so function wriiten below will be run once and stored in cache memory
@st.cache(allow_output_mutation = True) 

def load_model():
  model = tf.keras.models.load_model("/content/my_model.hdf5")
  return model

# loading our model
model = load_model  

# giving title toour webpage
st.write("""
        # COVID-19 detection
        """
         )
# uploading the image in web
file = st.file_uploader("Please upload an Ultrasound chest image", type = ["jpg","png"])
import cv2
from PIL import Image,ImageOps
import numpy as np

# a function which accepts image and the model and results prediction
def import_and_predict(image_data,model):

  size = (180,180)
  image = ImageOps.fit(image_data,size,Image.ANTIALIAS)
  img = np.asarray(image)
  img_reshape = img[np.newaxis,...]
  prediction = model.predict(img_reshape)

  return prediction

if file is None:
  st.text("please upload an image file")

else:
  image = Image.open(file)
  st.image(image,use_column_width=True)
  predictions = import_and_predict(image,model)
  class_names = ["Covid","Normal"]
  string = "Image most likely is:"+class_names[np.argmax(predictions)]
  st.success(string)
