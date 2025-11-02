import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configuration
IMG_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 50

# Disease categories and their data folders
DISEASES = {
    "Skin Lesion (Generic/Acne)": "data/acne_past",
    "Nail Psoriasis": "data/nail_psoriasis_past",
    "Dermatitis / Eczema": "data/dermatitis_past",
    "Stevens-Johnson Syndrome (SJS)": "data/sjs_past"
}

def load_images_from_folder(folder_path, img_size=IMG_SIZE):
    """
    Load all images from a folder and create binary masks.
    For now, we'll create synthetic masks based on basic thresholding.
    In a real scenario, you'd have ground truth masks.
    """
    images = []
    masks = []

    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist. Skipping...")
        return np.array(images), np.array(masks)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)

                # Create synthetic mask using basic segmentation
                # This is a placeholder - in practice you'd load ground truth masks
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                mask = cv2.resize(mask, img_size)
                masks.append(mask)

    return np.array(images), np.array(masks)

from tensorflow.keras.layers import UpSampling2D, Input

def create_segmentation_model(input_shape):
    """
    Create a simple U-Net like architecture for segmentation.
    """
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    # Decoder
    up4 = UpSampling2D((2, 2))(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(up4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)

    up5 = UpSampling2D((2, 2))(conv4)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(up5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)

    up6 = UpSampling2D((2, 2))(conv5)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)

    # Output
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv6)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model_for_disease(disease_name, data_folder):
    """
    Train a segmentation model for a specific disease.
    """
    print(f"\n=== Training model for {disease_name} ===")

    # Load images and masks
    images, masks = load_images_from_folder(data_folder)

    if len(images) == 0:
        print(f"No images found for {disease_name}. Skipping...")
        return None

    print(f"Loaded {len(images)} images for {disease_name}")

    # Normalize images
    images = images.astype('float32') / 255.0
    masks = masks.astype('float32') / 255.0
    masks = np.expand_dims(masks, axis=-1)  # Add channel dimension

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Create model
    model = create_segmentation_model((IMG_SIZE[0], IMG_SIZE[1], 3))

    # Train model
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        verbose=1
    )

    # Save model
    model_filename = f"models/{disease_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}_model.h5"
    os.makedirs('models', exist_ok=True)
    model.save(model_filename)
    print(f"Model saved as {model_filename}")

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{disease_name} - Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{disease_name} - Accuracy')
    plt.legend()

    plt.tight_layout()
    os.makedirs('models', exist_ok=True)
    plt.savefig(f'models/{disease_name.lower().replace(" ", "_")}_training_history.png')
    plt.close()  # Close the figure to avoid display issues

    return model

def main():
    """
    Train models for all diseases.
    """
    print("Starting model training for all diseases...")

    trained_models = {}

    for disease, folder in DISEASES.items():
        model = train_model_for_disease(disease, folder)
        if model is not None:
            trained_models[disease] = model

    print(f"\nTraining completed! Models trained: {list(trained_models.keys())}")

    # Summary
    print("\n=== Training Summary ===")
    for disease in DISEASES.keys():
        model_path = f"models/{disease.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}_model.h5"
        if os.path.exists(model_path):
            print(f"✓ {disease}: Model saved")
        else:
            print(f"✗ {disease}: No model (insufficient data)")

if __name__ == "__main__":
    main()
