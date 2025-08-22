import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np 

data_dir = 'C:\Users\Varsh\OneDrive\Desktop\music1a\Notes'

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),
    batch_size=16, 
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),
    batch_size=16,  
    class_mode='categorical',
    subset='validation'
)

X_train, y_train = [], []
X_test, y_test = [], []

for _ in range(train_generator.samples // train_generator.batch_size):
    img, label = next(train_generator)
    X_train.extend(img)
    y_train.extend(label)

for _ in range(validation_generator.samples // validation_generator.batch_size):
    img, label = next(validation_generator)
    X_test.extend(img)
    y_test.extend(label)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("Sample image shape:", X_train[0].shape)

plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[i])
    plt.title(np.argmax(y_train[i]))
    plt.axis('off')
plt.show()

X_train = X_train / 255.0
X_test = X_test / 255.0

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

'''plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()'''

val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

try:
    model.save('C:\\Users\\YASODHARA\\Downloads\\music\\music_note_classifier.h5')
    print("Model saved: music_note_classifier.h5")
except Exception as e:
    print("An error occurred while loading the model:", e)

from tensorflow.keras.models import load_model

try:
    model = load_model('C:\\Users\\YASODHARA\\Downloads\\music\\music_note_classifier.h5')
    print("Model Loaded: music_note_classifier.h5")
except Exception as e:
    print("An error occurred while loading the model:", e)

from tensorflow.keras.preprocessing import image
import numpy as np

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))  # Modelin giriş boyutu
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Modelin beklediği şekle getirin
    img_array /= 255.0  # Normalizasyon
    return img_array

def predict_image(model, img_path):
    img_array = prepare_image(img_path)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions, axis=1)
    return class_idx

class_labels = list(train_generator.class_indices.keys())

def get_class_label(class_idx):
    return class_labels[class_idx[0]]

img_path = 'C:\\Users\\YASODHARA\\Downloads\\trial whole.png'
class_idx = predict_image(model, img_path)
class_label = get_class_label(class_idx)

print(f'The predicted class is: {class_label}')
