import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.legacy.preprocessing.image import ImageDataGenerator

from plot_run_graph import plot_training_history

# Load Experimental Data (Simulated Dataset Example)
data = pd.read_csv('fluorescence_experiment_data.csv')  # CSV with bacterial load, treatment time, and outcome
tf.config.experimental_run_functions_eagerly(True)

# Extract Features & Labels
X = data[['fluorescence_intensity', 'treatment_time', 'drug_concentration']].values
y = data['bacterial_reduction'].values  # Target variable

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Model for Predicting Bacterial Reduction
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict & Evaluate
predictions = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f'Random Forest RMSE: {rmse:.2f}')

# # Create an ImageDataGenerator instance for data augmentation
# datagen = ImageDataGenerator(
#     rotation_range=30,     # Rotate images by up to 30 degrees
#     width_shift_range=0.2, # Shift width by up to 20%
#     height_shift_range=0.2,# Shift height by up to 20%
#     brightness_range=[0.6, 1.4],  # Stronger brightness variation
#     zoom_range=0.3,        # Zoom in/out by up to 30%
#     horizontal_flip=True   # Randomly flip images horizontally
# )

# Load & Preprocess Fluorescence Images for CNN
IMG_SIZE = 128
def load_images(folder):
    images, labels = [], []
    for label in ['cleared', 'persistent']:
        path = os.path.join(folder, label)
        for img_name in os.listdir(path):
            img = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
            images.append(img)
            labels.append(0 if label == 'persistent' else 1)  # Binary classification
    return np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1), np.array(labels)

X_img, y_img = load_images('fluorescence_images')
X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(X_img, y_img, test_size=0.2, random_state=42)

# # Fit the generator to the training images
# datagen.fit(X_train_img)

# Define & Train CNN Model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

cnn_model.add(Dropout(0.3))  # Drop 30% of neurons during training
optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)

cnn_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
history = cnn_model.fit(
    #datagen.flow(X_train_img, y_train_img, batch_size=32),  # Use data generator
    X_train_img,
    y_train_img,
    validation_data=(X_test_img, y_test_img),
    #steps_per_epoch = 100,
    epochs=60
)

plot_training_history(history, "run_results.png")

# Evaluate CNN Performance
cnn_predictions = (cnn_model.predict(X_test_img) > 0.5).astype('int')
accuracy = accuracy_score(y_test_img, cnn_predictions)
cm = confusion_matrix(y_test_img, cnn_predictions)

print(f'CNN Accuracy: {accuracy * 100:.2f}%')
sns.heatmap(cm, annot=True, fmt='d')
plt.show()
