import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import tkinter as tk
from tkinter import Canvas, Button

# Load the trained model
try:
    model = load_model('C:\\Users\\Varsh\\OneDrive\\Desktop\\music1a\\music_note_classifier_resnet.h5')
    print("Model Loaded: music_note_classifier.h5")
except Exception as e:
    print("An error occurred while loading the model:", e)

# Function to prepare the image
def prepare_image(img):
    img = img.resize((64, 64))
    img = img.convert("RGB")  # Convert image to RGB
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Function to predict the image
def predict_image(model, img):
    img_array = prepare_image(img)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions, axis=1)
    return class_idx

# Class labels
class_labels = ['Eight', 'Half', 'Quarter', 'Sixteenth', 'Whole']

# Function to get class label
def get_class_label(class_idx):
    return class_labels[class_idx[0]]

# Tkinter GUI
class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Music Note Classifier")
        self.canvas = Canvas(root, width=256, height=256, bg='white')
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)
        self.button_predict = Button(root, text="Predict", command=self.predict)
        self.button_predict.pack()
        self.button_clear = Button(root, text="Clear", command=self.clear)
        self.button_clear.pack()
        self.image = Image.new("RGB", (256, 256), "white")
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)
        self.draw.line([x1, y1, x2, y2], fill="black", width=5)

    def predict(self):
        # Convert the canvas drawing to an image
        self.image.save("temp.png")
        img = Image.open("temp.png")
        class_idx = predict_image(model, img)
        class_label = get_class_label(class_idx)
        print(f'The predicted class is: {class_label}')

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (256, 256), "white")
        self.draw = ImageDraw.Draw(self.image)

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
