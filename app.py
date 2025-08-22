from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import io
import base64

app = Flask(__name__)

# Load the trained model
model = load_model('C:\\Users\\Varsh\\OneDrive\\Desktop\\music1a\\music_note_classifier_resnet.h5')

# Class labels
class_labels = ['Eight', 'Half', 'Quarter', 'Sixteenth', 'Whole']

def prepare_image(img):
    img = img.resize((64, 64))
    img = img.convert("RGB")  # Convert image to RGB
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_image(model, img):
    img_array = prepare_image(img)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions, axis=1)
    return class_idx

def get_class_label(class_idx):
    return class_labels[class_idx[0]]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_data = data['image']
    img_data = base64.b64decode(img_data.split(',')[1])
    img = Image.open(io.BytesIO(img_data))
    class_idx = predict_image(model, img)
    class_label = get_class_label(class_idx)
    return jsonify({'class_label': class_label})

if __name__ == '__main__':
    app.run(debug=True)
