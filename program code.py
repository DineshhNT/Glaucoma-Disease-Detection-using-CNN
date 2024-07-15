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
validation_split=0.2                                                                 # 20% of the data will be used for validation ) 
train_generator = train_datagen.flow_from_directory(
 					dataset_path, 
        target_size=(img_height, img_width),
        batch_size=batch_size, 
        class_mode='binary', subset='training'          # set as training data ) 
validation_generator = train_datagen.flow_from_directory( 
            dataset_path, 
            target_size=(img_height, img_width), 
            batch_size=batch_size, 
            class_mode='binary', 
            subset='validation'                         # set as validation data )
 # Define the CNN model 
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3))) model.add(layers.MaxPooling2D((2, 2))) 
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
epochs = 10 history = model.fit( train_generator, epochs=epochs, validation_data=validation_generator ) # Save the trained model 
model.save("glaucoma_detection_model.h5")
