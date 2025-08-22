import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

# Clear session
from tensorflow.keras import backend as K
K.clear_session()

data_dir = 'C:\\Users\\Varsh\\OneDrive\\Desktop\\music1a\\Notes'

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),
    batch_size=8,  # Reduced batch size
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),
    batch_size=8,  # Reduced batch size
    class_mode='categorical',
    subset='validation'
)

def resnet_block(input_tensor, filters, kernel_size=3, stride=1):
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    
    if stride != 1 or input_tensor.shape[-1] != filters:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same')(input_tensor)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = input_tensor
    
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def build_resnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
    
    x = resnet_block(x, 64, stride=1)
    x = resnet_block(x, 64, stride=1)
    
    x = resnet_block(x, 128, stride=2)
    x = resnet_block(x, 128, stride=1)
    
    x = resnet_block(x, 256, stride=2)
    x = resnet_block(x, 256, stride=1)
    
    x = resnet_block(x, 512, stride=2)
    x = resnet_block(x, 512, stride=1)
    
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

input_shape = (64, 64, 3)
num_classes = 5
model = build_resnet(input_shape, num_classes)

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
    model.save('C:\\Users\\YASODHARA\\Downloads\\music\\music_note_classifier_resnet.h5')
    print("Model saved: music_note_classifier_resnet.h5")
except Exception as e:
    print("An error occurred while saving the model:", e)

from tensorflow.keras.models import load_model

try:
    model = load_model('C:\\Users\\YASODHARA\\Downloads\\music\\music_note_classifier_resnet.h5')
    print("Model Loaded: music_note_classifier_resnet.h5")
except Exception as e:
    print("An error occurred while loading the model:", e)

from tensorflow.keras.preprocessing import image

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
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
