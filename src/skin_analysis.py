import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model 

# --- NEW IMPORTS: Import the fallback logic from the new files ---
from src.analysis_methods.generic import generic_lesion_fallback
from src.analysis_methods.nail_psoriasis import nail_psoriasis_fallback
from src.analysis_methods.diffuse import diffuse_lesion_fallback
from src.spin_analysis import scoliosis_fallback
# ---------------------------------------------------------------

# --- 1. CONFIGURATION ---
PAST_IMAGE_PATH = 'data/past_lesion.jpeg'
NEW_IMAGE_PATH = 'data/new_lesion.jpeg'
INPUT_SIZE = (256, 256) 

# --- Model Path Dictionary (Remains the same) ---
MODEL_PATHS = {
    "Skin Lesion (Generic/Acne)": 'models/lesion_segmentation_model.h5',
    "Nail Psoriasis": 'models/nail_psoriasis_model.h5',
    "Dermatitis / Eczema": 'models/dermatitis_model.h5',
    "Stevens-Johnson Syndrome (SJS)": 'models/sjs_model.h5'
}
segmentation_models = {}
segmentation_model = None


def load_segmentation_model(disease):
    """
    Loads the correct pre-trained Keras model for the selected disease.
    (This function remains unchanged from your last version)
    """
    global segmentation_models
    model_path = MODEL_PATHS.get(disease)
    # ... (loading logic remains the same) ...
    if model_path is None:
        print(f"[ERROR] No model path defined for disease: {disease}")
        return None
    if disease in segmentation_models:
        print(f"[STATUS] Model for {disease} already loaded.")
        return segmentation_models[disease]
    if not os.path.exists(model_path):
        print(f"[STATUS] Model file not found at: {model_path}")
        print(f"[STATUS] Proceeding with {disease}-specific fallback.")
        return None
    try:
        print(f"[STATUS] Attempting to load model for {disease} from {model_path}...")
        model = load_model(model_path, compile=False)
        segmentation_models[disease] = model
        print(f"[STATUS] Model for {disease} loaded successfully.")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model for {disease}: {e}")
        print(f"[STATUS] Proceeding with {disease}-specific fallback.")
        return None


def segment_lesion(image, disease="Skin Lesion (Generic/Acne)"):
    """
    Segmentation function using either the Deep Learning model or a Disease-specific fallback.
    """
    model = load_segmentation_model(disease) 

    if model is not None:
        # --- DEEP LEARNING SEGMENTATION PATH (Remains the same) ---
        print(f"Using Deep Learning Segmentation for: {disease}.")
        resized_img = cv2.resize(image, INPUT_SIZE)
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        
        # Normalize and prepare input batch
        input_data = np.expand_dims(rgb_img, axis=0) / 255.0

        # Prediction
        prediction = model.predict(input_data, verbose=0)[0] 

        # Convert probability map to binary mask
        mask = (prediction[..., 0] > 0.5).astype(np.uint8) 
        
        # Resize mask back to original size
        original_size = (image.shape[1], image.shape[0])
        final_mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
        
        return final_mask * 255, 0 
        
    else:
        # --- DISEASE-SPECIFIC FALLBACK PATH (Refactored to call imported functions) ---
        print(f"Using rule-based segmentation fallback for: {disease}.")
        
        if disease == "Nail Psoriasis":
            mask, count = nail_psoriasis_fallback(image)

        elif disease in ["Dermatitis / Eczema", "Stevens-Johnson Syndrome (SJS)"]:
            mask, count = diffuse_lesion_fallback(image)

        elif disease == "Scoliosis":
            mask, count = scoliosis_fallback(image)

        else: # Default: "Skin Lesion (Generic/Acne)"
            mask, count = generic_lesion_fallback(image)
            
        return mask, count

# ---------------------------------------------------------------------
# --- MEASUREMENT & MAIN FUNCTIONS (Remains the same) ---
# ---------------------------------------------------------------------

def calculate_lesion_area(mask):
    """Calculates the area of the segmented lesion in pixels."""
    area = np.sum(mask > 0)
    return area

# ... (main function remains unchanged) ...

if __name__ == '__main__':
    # ... (main logic remains unchanged) ...
    pass