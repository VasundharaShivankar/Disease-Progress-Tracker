import os
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from pymongo import MongoClient
import urllib.parse

# Load environment variables from .env file
load_dotenv()

# Load environment variables for MongoDB Atlas connection
# You can store either a full URI in MONGODB_CONNECTION_STRING or store user, pass, host separately.
CONNECTION_STRING = os.environ.get("MONGODB_CONNECTION_STRING")
DATABASE_NAME = os.environ.get("MONGODB_DATABASE_NAME")

# Optional separate credentials handling (uncomment if you prefer pieces)
# MONGO_USER = os.environ.get("MONGODB_USER")
# MONGO_PASS = os.environ.get("MONGODB_PASS")
# MONGO_HOST = os.environ.get("MONGODB_HOST")  # e.g. cluster0.dvkg9pv.mongodb.net
# if MONGO_USER and MONGO_PASS and MONGO_HOST:
#     CONNECTION_STRING = (
#         f"mongodb+srv://{urllib.parse.quote_plus(MONGO_USER)}:"
#         f"{urllib.parse.quote_plus(MONGO_PASS)}@{MONGO_HOST}/?retryWrites=true&w=majority"
#     )

# Basic checks
if not CONNECTION_STRING:
    raise ValueError("MONGODB_CONNECTION_STRING environment variable is not set.")
if not DATABASE_NAME:
    raise ValueError("MONGODB_DATABASE_NAME environment variable is not set.")

# OPTIONAL: quick debug (never print secrets in production)
masked = CONNECTION_STRING
if "@" in CONNECTION_STRING and ":" in CONNECTION_STRING.split("@")[0]:
    # mask credentials for debug
    pre, post = CONNECTION_STRING.split("@", 1)
    if ":" in pre:
        username_pass = pre.split("//", 1)[1]
        masked = CONNECTION_STRING.replace(username_pass, "****:****")

print(f"Using DB: {DATABASE_NAME}, connecting to: {masked}")

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

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    pil_mask = Image.fromarray(mask)
    buffer = BytesIO()
    pil_mask.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def upload_folder_to_mongo(folder_path, collection_name, category):
    """Upload all images from a folder to MongoDB collection."""
    try:
        # Use a small serverSelectionTimeoutMS so bad connections fail fast
        client = MongoClient(CONNECTION_STRING, serverSelectionTimeoutMS=5000)
        # Test the connection
        client.admin.command('ping')
        print("Successfully connected to MongoDB Atlas.")
    except Exception as e:
        print(f"Failed to connect to MongoDB Atlas: {e}")
        return

    db = client[DATABASE_NAME]
    collection = db[collection_name]

    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist. Skipping...")
        client.close()
        return

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    uploaded_count = 0

    for filename in files:
        image_path = os.path.join(folder_path, filename)

        try:
            image_b64 = encode_image_to_base64(image_path)
            mask_b64 = create_synthetic_mask(image_path)

            document = {
                "filename": filename,
                "image": image_b64,
                "mask": mask_b64,
                "category": category,
                "disease": collection_name
            }

            collection.insert_one(document)
            uploaded_count += 1
            print(f"Uploaded {filename} to {collection_name}")

        except Exception as e:
            print(f"Failed to upload {filename}: {e}")

    print(f"Uploaded {uploaded_count} images to {collection_name}")
    client.close()

def main():
    print("Starting upload to MongoDB Atlas...")

    for folder, collection in DISEASE_COLLECTIONS.items():
        folder_path = os.path.join("data", folder)
        upload_folder_to_mongo(folder_path, collection, "past")

    for folder, collection in DISEASE_COLLECTIONS.items():
        present_folder = folder.replace("_past", "_present")
        folder_path = os.path.join("data", present_folder)
        if os.path.exists(folder_path):
            upload_folder_to_mongo(folder_path, collection, "present")

    print("Upload complete!")

if __name__ == "__main__":
    main()
