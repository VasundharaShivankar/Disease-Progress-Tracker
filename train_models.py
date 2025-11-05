import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pymongo import MongoClient
import base64
from io import BytesIO
from PIL import Image

# Configuration
IMG_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 50

# MongoDB Atlas connection
CONNECTION_STRING = "mongodb+srv://vasundharashivankar179:DyRcGbNUJtRWjVPk@cluster0.dvkg9pv.mongodb.net/"
DATABASE_NAME = "nutritional_assessment"

# Disease categories and their MongoDB collections
DISEASES = {
    "Skin Lesion (Generic/Acne)": "skin_lesion_generic_acne",
    "Nail Psoriasis": "nail_psoriasis",
    "Dermatitis / Eczema": "dermatitis_eczema",
    "Stevens-Johnson Syndrome (SJS)": "stevens_johnson_syndrome"
}

def generate_synthetic_data(num_samples=100, img_size=IMG_SIZE):
    """
    Generate synthetic images and masks for demonstration when real data is unavailable.
    """
    images = []
    masks = []

    for i in range(num_samples):
        # Generate synthetic skin-like image
        img = np.random.randint(200, 255, (img_size[0], img_size[1], 3), dtype=np.uint8)

        # Add some random lesions (dark spots)
        num_lesions = np.random.randint(1, 5)
        for _ in range(num_lesions):
            center_x = np.random.randint(50, img_size[0]-50)
            center_y = np.random.randint(50, img_size[1]-50)
            radius = np.random.randint(10, 30)

            # Draw lesion
            cv2.circle(img, (center_x, center_y), radius, (100, 80, 70), -1)

        images.append(img)

        # Create corresponding mask
        mask = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
        # In real scenario, this would be ground truth
        # For synthetic, create simple circular masks
        for _ in range(num_lesions):
            center_x = np.random.randint(50, img_size[0]-50)
            center_y = np.random.randint(50, img_size[1]-50)
            radius = np.random.randint(10, 30)
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)

        masks.append(mask)

    return np.array(images), np.array(masks)

def decode_base64_image(base64_string):
    """Decode base64 string to PIL Image."""
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return np.array(image)

def decode_base64_mask(base64_string):
    """Decode base64 string to mask array."""
    mask_data = base64.b64decode(base64_string)
    mask = Image.open(BytesIO(mask_data))
    return np.array(mask)

def load_images_from_mongodb(collection_name, img_size=IMG_SIZE):
    """
    Load all images and masks from MongoDB collection.
    If no data exists, generate synthetic data for demonstration.
    """
    images = []
    masks = []

    try:
        client = MongoClient(CONNECTION_STRING)
        db = client[DATABASE_NAME]
        collection = db[collection_name]

        # Query for 'past' category images (for training baseline models)
        documents = list(collection.find({"category": "past"}))

        if len(documents) == 0:
            print(f"No documents found in {collection_name}. Generating synthetic data...")
            client.close()
            return generate_synthetic_data(num_samples=50, img_size=img_size)

        for doc in documents:
            try:
                # Decode image
                img_b64 = doc.get('image')
                if img_b64:
                    img = decode_base64_image(img_b64)
                    img = cv2.resize(img, img_size)
                    if img.shape[-1] == 4:  # RGBA to RGB
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                    elif len(img.shape) == 2:  # Grayscale to RGB
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    images.append(img)

                # Decode mask
                mask_b64 = doc.get('mask')
                if mask_b64:
                    mask = decode_base64_mask(mask_b64)
                    mask = cv2.resize(mask, img_size)
                    if len(mask.shape) == 3 and mask.shape[-1] > 1:
                        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                    masks.append(mask)
                else:
                    # If no mask, create synthetic one
                    if len(images) > 0:
                        gray = cv2.cvtColor(images[-1], cv2.COLOR_RGB2GRAY)
                        _, synth_mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                        masks.append(synth_mask)

            except Exception as e:
                print(f"Error processing document {doc.get('filename', 'unknown')}: {e}")
                continue

        client.close()

    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        print("Generating synthetic data...")
        return generate_synthetic_data(num_samples=50, img_size=img_size)

    if len(images) < 10:
        print(f"Only {len(images)} images found. Supplementing with synthetic data...")
        synth_images, synth_masks = generate_synthetic_data(num_samples=50, img_size=img_size)
        images = np.concatenate([images, synth_images])
        masks = np.concatenate([masks, synth_masks])

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

def train_model_for_disease(disease_name, collection_name):
    """
    Train a segmentation model for a specific disease.
    """
    print(f"\n=== Training model for {disease_name} ===")

    # Load images and masks from MongoDB
    images, masks = load_images_from_mongodb(collection_name)

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
    # Fix filename to avoid special characters
    safe_filename = disease_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    plt.savefig(f'models/{safe_filename}_training_history.png')
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
