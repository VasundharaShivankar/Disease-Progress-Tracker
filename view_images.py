import os
from dotenv import load_dotenv
from pymongo import MongoClient
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()
CONNECTION_STRING = os.environ.get("MONGODB_CONNECTION_STRING")
DATABASE_NAME = os.environ.get("MONGODB_DATABASE_NAME")

if not CONNECTION_STRING or not DATABASE_NAME:
    raise ValueError("Environment variables not set.")

# Connect to MongoDB
client = MongoClient(CONNECTION_STRING)
db = client[DATABASE_NAME]

def view_images_from_collection(collection_name, limit=None):
    """Fetch and display images from a MongoDB collection."""
    collection = db[collection_name]
    if limit:
        documents = collection.find().limit(limit)
    else:
        documents = collection.find()

    images = []
    filenames = []
    for doc in documents:
        img_data = base64.b64decode(doc['image'])
        img = Image.open(BytesIO(img_data))
        images.append(img)
        filenames.append(doc.get('filename', 'Unknown'))

    if images:
        num_images = len(images)
        cols = min(5, num_images)  # Max 5 columns
        rows = (num_images + cols - 1) // cols  # Ceiling division

        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))

        # Handle axes flattening properly
        if rows == 1 and cols == 1:
            axes = [axes]  # Single subplot
        else:
            axes = axes.flatten()

        for i, (ax, img, fname) in enumerate(zip(axes, images, filenames)):
            ax.imshow(img)
            ax.set_title(fname, fontsize=8)
            ax.axis('off')

        # Hide unused subplots
        for i in range(num_images, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f"All Images from {collection_name} ({num_images} images)")
        plt.tight_layout()
        plt.show()
    else:
        print(f"No images found in {collection_name}")

if __name__ == "__main__":
    collections = ["skin_lesion_generic_acne", "nail_psoriasis", "dermatitis_eczema"]
    for col in collections:
        view_images_from_collection(col)
    client.close()
