import os
import cv2
import numpy as np
from pymongo import MongoClient
import base64
from io import BytesIO
from PIL import Image

# MongoDB Atlas connection
CONNECTION_STRING = "mongodb+srv://vasundharashivankar179:DyRcGbNUJtRWjVPk@cluster0.dvkg9pv.mongodb.net/"
DATABASE_NAME = "nutritional_assessment"

# Disease mappings to collections
DISEASE_COLLECTIONS = {
    "acne_past": "skin_lesion_generic_acne",
    "nail_psoriasis_past": "nail_psoriasis",
    "dermatitis_past": "dermatitis_eczema",
    "sjs_past": "stevens_johnson_syndrome"
}

def encode_image_to_base64(image_path):
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_synthetic_mask(image_path):
    """Create a synthetic mask for demonstration (in real scenario, use ground truth)."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Simple synthetic mask creation (placeholder)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Convert to base64
    pil_mask = Image.fromarray(mask)
    buffer = BytesIO()
    pil_mask.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def upload_folder_to_mongo(folder_path, collection_name, category):
    """Upload all images from a folder to MongoDB collection."""
    client = MongoClient(CONNECTION_STRING)
    db = client[DATABASE_NAME]
    collection = db[collection_name]

    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist. Skipping...")
        return

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    uploaded_count = 0

    for filename in files:
        image_path = os.path.join(folder_path, filename)

        try:
            # Encode image
            image_b64 = encode_image_to_base64(image_path)

            # Create synthetic mask (replace with real masks if available)
            mask_b64 = create_synthetic_mask(image_path)

            # Create document
            document = {
                "filename": filename,
                "image": image_b64,
                "mask": mask_b64,
                "category": category,  # 'past' or 'present'
                "disease": collection_name
            }

            # Insert into collection
            collection.insert_one(document)
            uploaded_count += 1
            print(f"Uploaded {filename} to {collection_name}")

        except Exception as e:
            print(f"Failed to upload {filename}: {e}")

    print(f"Uploaded {uploaded_count} images to {collection_name}")
    client.close()

def main():
    """Upload all local data to MongoDB Atlas."""
    print("Starting upload to MongoDB Atlas...")

    # Upload past images
    for folder, collection in DISEASE_COLLECTIONS.items():
        folder_path = os.path.join("data", folder)
        upload_folder_to_mongo(folder_path, collection, "past")

    # Upload present images if available (assuming similar structure)
    for folder, collection in DISEASE_COLLECTIONS.items():
        present_folder = folder.replace("_past", "_present")
        folder_path = os.path.join("data", present_folder)
        if os.path.exists(folder_path):
            upload_folder_to_mongo(folder_path, collection, "present")

    print("Upload complete!")

if __name__ == "__main__":
    main()
