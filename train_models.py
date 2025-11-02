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

def load_images_from_folder(folder_path, img_size=IMG_SIZE):
    """
    Load all images from a folder and create binary masks.
    If no real data exists, generate synthetic data for demonstration.
    """
    images = []
    masks = []

    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist. Generating synthetic data...")
        return generate_synthetic_data(num_samples=50, img_size=img_size)

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(files) == 0:
        print(f"No images found in {folder_path}. Generating synthetic data...")
        return generate_synthetic_data(num_samples=50, img_size=img_size)

    for filename in files:
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
