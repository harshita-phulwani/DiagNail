from keras.models import load_model
import joblib
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.applications.vgg16 import VGG16
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load VGG16 model
vgg16_model = VGG16(weights='diagnail.h5', include_top=False, input_shape=(256, 256, 3))# load_model('diagnail.h5',  compile=False)

# Load Random Forest model
rf_model = joblib.load('diagnail.joblib')

# Preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((256, 256))  # Resize image to VGG16 input size
    img = np.array(img)
    img = img / 255.0  # Normalize pixel values
    return img

# Extract features with VGG16
def extract_features(image):
    # Assuming your VGG16 model's input shape is (None, 224, 224, 3)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    features = vgg16_model.predict(image)
    return features

# Predict with Random Forest
def predict_with_rf(features):
    return rf_model.predict(features)

# Image path provided by the user
image_path = 'img1.jpeg'

# Preprocess the image
preprocessed_image = preprocess_image(image_path)

# Extract features using VGG16
image_features = extract_features(preprocessed_image)

# Predict using Random Forest
prediction = predict_with_rf(image_features)

# Post-processing
# Here you might convert numerical prediction into class labels or perform any other necessary transformations

print("Prediction:", prediction)
