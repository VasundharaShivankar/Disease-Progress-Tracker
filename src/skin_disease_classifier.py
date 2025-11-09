import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Skin Disease Classification Module
# Based on the skinfoolder2 model from GitHub

MODEL_PATH = 'skinmodel_vgg16.h5'
IMG_SIZE = (224, 224)

# Global model variable
skin_model = None

# Class labels (ensure the order matches your training)
class_labels = ['acne', 'hyperpigmentation', 'nail_psoriasis', 'sjsten', 'vitiligo']

# Detailed info for each class
disease_info = {
    'acne': {
        'explanation': "Acne may be caused by hormonal changes, excess oil production, or dietary factors like high sugar or dairy intake.",
        'tips': [
            "Drink at least 2-3 liters of water daily.",
            "Include foods rich in omega-3s like walnuts and flaxseeds.",
            "Avoid oily, fried, and sugary foods.",
            "Wash your face twice daily with a mild cleanser.",
            "Use non-comedogenic skincare products."
        ],
        'advice': "If acne becomes severe or painful, please consult a dermatologist for professional treatment."
    },
    'hyperpigmentation': {
        'explanation': "Hyperpigmentation often results from sun exposure, inflammation, or hormonal changes.",
        'tips': [
            "Apply sunscreen daily with SPF 30 or more.",
            "Eat antioxidant-rich foods like berries and leafy greens.",
            "Avoid picking or scratching the skin.",
            "Use products with Vitamin C or niacinamide.",
            "Stay hydrated to improve skin healing."
        ],
        'advice': "Persistent pigmentation may require a dermatologist's evaluation."
    },
    'nail_psoriasis': {
        'explanation': "Nail psoriasis is linked to immune dysfunction and can be worsened by stress or nutrient deficiencies.",
        'tips': [
            "Take biotin-rich foods like eggs, almonds, and sweet potatoes.",
            "Keep nails trimmed and clean.",
            "Avoid nail injuries and harsh chemicals.",
            "Apply moisturizers or medicated nail creams.",
            "Manage stress through yoga or meditation."
        ],
        'advice': "For visible damage or pain, please consult a dermatologist or rheumatologist."
    },
    'sjsten': {
        'explanation': "Stevens-Johnson Syndrome (SJS) and Toxic Epidermal Necrolysis (TEN) are serious skin reactions, often to medications or infections.",
        'tips': [
            "Avoid self-medication, especially antibiotics or NSAIDs.",
            "Boost immunity with fruits, vegetables, and multivitamins.",
            "Hydrate well and maintain oral hygiene.",
            "Seek immediate help if rashes spread rapidly or are painful.",
            "Be aware of any new medication reactions."
        ],
        'advice': "This condition is a medical emergency — immediate hospitalization is necessary. Consult a doctor without delay."
    },
    'vitiligo': {
        'explanation': "Vitiligo is an autoimmune condition where pigment-producing cells are lost.",
        'tips': [
            "Eat foods rich in antioxidants (e.g., citrus fruits, green tea).",
            "Ensure good Vitamin D intake through sunlight or supplements.",
            "Use sunscreen to protect depigmented areas.",
            "Avoid stress — it may worsen the spread.",
            "Consult on PUVA or laser therapy options if needed."
        ],
        'advice': "For personalized treatment and slowing spread, consult a dermatologist."
    }
}

def load_skin_model():
    """
    Load the skin disease classification model.
    """
    global skin_model
    if skin_model is None:
        try:
            skin_model = load_model(MODEL_PATH, compile=False)
            print("[STATUS] Skin disease classification model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load skin model: {e}")
            return None
    return skin_model

def classify_skin_disease(image):
    """
    Classify skin disease from image using the pre-trained model.

    Args:
        image: OpenCV image array (BGR format)

    Returns:
        dict: Classification results with prediction, confidence, explanation, tips, and advice
    """
    model = load_skin_model()
    if model is None:
        return {
            'prediction': 'Error',
            'confidence': '0%',
            'explanation': 'Model failed to load.',
            'tips': [],
            'advice': 'Please try again later.'
        }

    try:
        # Convert BGR to RGB and resize
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        img = pil_img.resize(IMG_SIZE)

        # Preprocess image
        img_array = np.array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        class_index = int(np.argmax(predictions[0]))
        class_label = class_labels[class_index]
        confidence = float(np.max(predictions[0]))

        # Get disease information
        info = disease_info.get(class_label, {
            'explanation': "No information available.",
            'tips': [],
            'advice': ""
        })

        return {
            'prediction': class_label,
            'confidence': f"{confidence*100:.2f}%",
            'explanation': info['explanation'],
            'tips': info['tips'],
            'advice': info['advice']
        }

    except Exception as e:
        print(f"[ERROR] Skin classification failed: {e}")
        return {
            'prediction': 'Error',
            'confidence': '0%',
            'explanation': f'Classification failed: {str(e)}',
            'tips': [],
            'advice': 'Please try again with a different image.'
        }
