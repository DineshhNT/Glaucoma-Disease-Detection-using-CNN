# Glaucoma Detection using Convolutional Neural Networks

## Overview
This project employs a Convolutional Neural Network (CNN) to assist in the detection of glaucoma from eye images. Glaucoma is a disease affecting the vision of the human eye, and early detection is crucial for effective treatment. The CNN helps differentiate patterns between glaucoma and non-glaucoma images using a hierarchical structure.

## Concept
- **Convolutional Neural Network (CNN)**: Utilized for detecting glaucoma.
- **Disease**: Glaucoma, related to human eye vision.
- **Pattern Differentiation**: CNN distinguishes patterns in glaucoma vs. non-glaucoma images.
- **Model Architecture**: Six layers used for evaluation.

## Steps
1. **Collect Image Data**: Gather a labeled dataset of eye images (glaucoma and non-glaucoma).
2. **Preprocess the Data**: Normalize, augment, and split data into training, validation, and test sets.
3. **Model the Architecture**: Define the CNN structure.
4. **Train the Dataset Model**: Fit the model using the training data.
5. **Validation and Testing**: Validate and test the model with separate sets.
6. **Model Interpretation and Deployment**: Analyze and deploy the trained model.

## Requirements
- TensorFlow
- Keras
- scikit-learn
- os

## Installation
Install the required libraries using pip:
```bash
pip install tensorflow keras scikit-learn
```

## Usage

### 1. Dataset Preparation
Ensure your dataset is organized in a directory with subdirectories for each class (glaucoma and non-glaucoma). Update the `dataset_path` variable with the path to your dataset.

### 2. Preprocess the Data
Use `ImageDataGenerator` for data augmentation and normalization.

### 3. Define the CNN Model
Create the CNN model with six layers for evaluation.

### 4. Train the Model
Train the model with the training data and validate using the validation set.

### 5. Save the Trained Model
Save the trained model for future use.

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os

# Set the path to your dataset
dataset_path = "/content/drive/MyDrive/DATASET DL"

# Define image dimensions and batch size
img_height, img_width = 128, 128
batch_size = 32

# Use ImageDataGenerator for data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% of the data will be used for validation
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'  # set as training data
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'  # set as validation data
)

# Define the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

# Save the trained model
model.save("glaucoma_detection_model.h5")
```

### 6. Model Output
Ensure the model outputs appropriate logs during training to monitor performance.

### Common Issues and Fixes
- **Incorrect Loss Values**: Verify data preprocessing steps and model architecture.
- **Accuracy Stagnation**: Experiment with different model architectures, data augmentation techniques, and hyperparameters.

This README provides a comprehensive overview of the project, instructions for usage, and guidance for future improvements. Adjust the details based on your specific project requirements and data.
