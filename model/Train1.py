import os
from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import cv2
from matplotlib.image import imread
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import glob
import PIL
import random
from tqdm import tqdm
import sys
import math

# Add the root project directory to Python's search path
sys.path.append(r"C:\Users\ROG\Desktop\Machine Learning\ILC early detection\Visualization")
from Vis import dicom_cleaned_data, Data_cleaning_1, Data_cleaning_2, non_cancer_imgs, cancer_imgs

print("2")

# Randomly sample images from two lists
MAX_IMAGES = 200000
some_can_img = cancer_imgs.copy()
num_can = len(some_can_img)
num_non_needed = MAX_IMAGES - num_can
num_non_available = len(non_cancer_imgs)
num_non_to_use = min(num_non_available, num_non_needed)
some_non_img = random.sample(non_cancer_imgs, num_non_to_use)

print(f"Using {num_can} cancer images and {num_non_to_use} non-cancer images, total = {num_can + num_non_to_use}")

target_size = (224, 224)
non_img_arr = []
can_img_arr = []

print("Loading non-cancer image paths...")
for img in tqdm(some_non_img):
    non_img_arr.append([img, 0])

print("Loading cancer image paths...")
for img in tqdm(some_can_img):
    can_img_arr.append([img, 1])

# Use dataframe-based generator to avoid memory load
image_paths = [img for img, _ in non_img_arr + can_img_arr]
labels = [label for _, label in non_img_arr + can_img_arr]

df = pd.DataFrame({
    'filename': image_paths,
    'class': labels
})
df['class'] = df['class'].astype(str)

# Split dataframe
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class'])

# Define generators
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_datagen = datagen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='class',
    target_size=target_size,
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

test_datagen = datagen.flow_from_dataframe(
    test_df,
    x_col='filename',
    y_col='class',
    target_size=target_size,
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    min_delta=1e-7,
    restore_best_weights=True,
)

plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_delta=1e-7,
    cooldown=0,
    verbose=1
)

# EfficientNetB0 Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train model
history = model.fit(
    train_datagen,
    validation_data=test_datagen,
    epochs=25,
    callbacks=[early_stopping, plateau]
)

# Evaluate
model.evaluate(test_datagen)

# Prediction and confusion matrix
Y_pred = model.predict(test_datagen, steps=math.ceil(test_datagen.samples / test_datagen.batch_size))
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = test_datagen.classes

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="BuPu", linecolor="gray", fmt='.1f', ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Accuracy Plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Loss Plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Show sample prediction
index = 90
sample_path = test_df.iloc[index]['filename']
sample_img = cv2.imread(sample_path)
sample_img_resized = cv2.resize(sample_img, target_size)
sample_input = np.expand_dims(sample_img_resized / 255.0, axis=0)
predicted_class_index = model.predict(sample_input)[0].argmax()
true_class_index = int(test_df.iloc[index]['class'])

class_labels = {0: 'Non-Cancerous', 1: 'Cancerous'}
calcification_types = {0: 'No Calcification', 1: 'PLEOMORPHIC'}

plt.title('Test Image')
plt.imshow(sample_img_resized)
predicted_label = class_labels[predicted_class_index]
true_label = class_labels[true_class_index]
calcification_type = calcification_types.get(predicted_class_index, "Unknown")

print('Predicted Diagnosis:', predicted_label)
print('Calcification Type:', calcification_type)
print('True Diagnosis:', true_label)

# Save model
def get_next_model_filename(base_name='CNN_model', extension='.h5', directory='.'):
    i = 1
    while os.path.exists(os.path.join(directory, f"{base_name}_{i}{extension}")):
        i += 1
    return os.path.join(directory, f"{base_name}_{i}{extension}")

next_model_path = get_next_model_filename()
model.save(next_model_path)
print(f"Model saved as: {next_model_path}")
