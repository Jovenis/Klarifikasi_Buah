import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('my_model2.hdf5')
    return model

with st.spinner('Model is being loaded..'):
    model = load_model()

st.write("""
         # Fruit Classification
         """
         )

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)

class_names = ["class_1", "class_2", "class_3"]  # Replace with your actual class names

def import_and_predict(image_data, model):
    size = (180, 180)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis, ...]
    
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    st.write(predictions)  # Fixed typo here (changed 'prediction' to 'predictions')
    st.write(score)
    
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
