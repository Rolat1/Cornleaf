import streamlit as st
from PIL import Image
from keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np
import joblib

# Load the saved SVM model
model_filename = 'svm_model.joblib'
loaded_model = joblib.load(model_filename)


st.markdown('<h1 style="color:gray;">Corn Leaf Diseases Detection Model</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="color:gray;">The image classification model classifies corn leaf image into following categories:</h2>', unsafe_allow_html=True)
st.markdown('<h3 style="color:gray;"> gray spot,  common rust, healthy, northern leaf blight</h3>', unsafe_allow_html=True)

upload= st.file_uploader('Insert corn leaf image ', type=['png','jpg'])
c1, c2= st.columns(2)

if upload is not None:
  im= Image.open(upload)
  
  img= im.resize((224, 224))
  image= np.asarray(img)
  img= preprocess_input(image)
  img= np.expand_dims(img, 0)
  c1.header('Input Image')
  c1.image(im)
  # c1.write(img.shape)


  # Load the VGG16 model
  weights_path = '/Users/macbookpro/Downloads/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

  vgg16_model = VGG16(weights=weights_path, include_top=False)

  # Extract features using the VGG16 model
  features = vgg16_model.predict(img)

  # Global Average Pooling to reduce the number of features to 512
  img_features = np.mean(features, axis=(1, 2))

  # Flatten the features to make it 1D
  img_features = img_features.flatten()

  # Make predictions using the SVM model
  prediction = loaded_model.predict([img_features])

  # Decode the prediction back to class labels
  predicted_class = np.argmax(prediction)

  c2.header('Output')
  c2.subheader('Predicted class :')
  # c2.write(prediction)
  if predicted_class == 0:
    c2.write("The leaf has gray spots")
  elif predicted_class == 1:
    c2.write("The leaf has common rust disease")
  elif predicted_class == 2:
    c2.write("The leaf is healthy")
  else:
    c2.write("The leaf has nothern leaf blight disease")
