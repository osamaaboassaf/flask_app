import os
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


# Create a Flask app
app = Flask(__name__)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load the trained model
model = load_model('my_model.h5')  


# Preprocess the image to the format your model expects
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((32, 32))  # Resize to match model input size (32x32)
    img = np.array(img) / 255.0   # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img.reshape(len(img), 3072)
    return img


# Route for the home page
@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/index')
def index():
    return render_template('index.html')

# Route to handle file upload and model prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the file to the upload folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess the image and make prediction
        img = preprocess_image(filepath)
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=-1)
        LABEL_NAMES = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
        predicted_class=LABEL_NAMES[int(predicted_class)]

        # Return the result along with the image
        return render_template('index.html', prediction=predicted_class, image_url=filepath)

if __name__ == '__main__':
    app.run(debug=True)
