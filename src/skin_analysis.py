import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model 

# --- 1. CONFIGURATION ---
PAST_IMAGE_PATH = 'data/past_lesion.jpeg'
NEW_IMAGE_PATH = 'data/new_lesion.jpeg'
MODEL_PATH = 'models/segmentation_model.h5'
INPUT_SIZE = (256, 256) # Standard size for many U-Net models

# Global variable to store the loaded model
segmentation_model = None

def load_segmentation_model():
    """Loads the pre-trained Keras model from the file system."""
    global segmentation_model
    
    # Check 1: Check if the file exists before attempting the resource-intensive load
    if not os.path.exists(MODEL_PATH):
        print(f"[STATUS] Model file not found at: {MODEL_PATH}")
        print("[STATUS] Proceeding with disease-specific segmentation fallbacks.")
        return None

    # Check 2: Load only if not already loaded and file exists
    if segmentation_model is None:
        try:
            print(f"[STATUS] Model file found. Attempting to load model...")
            # Set compile=False if you only need inference and not training/compiling
            segmentation_model = load_model(MODEL_PATH, compile=False)
            print("[STATUS] Model loaded successfully.")
        except Exception as e:
            # Handles errors like corrupted file or incorrect format
            print(f"[ERROR] Failed to load model: {e}")
            print("[STATUS] Proceeding with disease-specific segmentation fallbacks.")
            return None
            
    return segmentation_model

def segment_lesion(image, disease="Skin Lesion (Generic/Acne)"): # <-- ADD disease argument
    """
    Segmentation function using either the Deep Learning model or a Disease-specific fallback.
    
    RETURNS: (mask_image, lesion_count)
    """
    model = load_segmentation_model()

    if model is not None:
        # --- DEEP LEARNING SEGMENTATION PATH ---
        print("Using Deep Learning Segmentation.")
        resized_img = cv2.resize(image, INPUT_SIZE)
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        
        # Normalize and prepare input batch
        input_data = np.expand_dims(rgb_img, axis=0) / 255.0

        # Prediction
        prediction = model.predict(input_data, verbose=0)[0] 

        # Convert probability map to binary mask
        mask = (prediction > 0.5).astype(np.uint8) 
        
        # Resize mask back to original size
        original_size = (image.shape[1], image.shape[0])
        final_mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
        
        # Scale mask back to 0-255
        return final_mask * 255, 0 
        
    else:
        # --- DISEASE-SPECIFIC FALLBACK PATH ---
        if disease == "Nail Psoriasis":
            print("Using NAIL PSORIASIS specialized fallback.")
            mask, count = nail_psoriasis_fallback(image)
        
        elif disease in ["Dermatitis / Eczema", "Stevens-Johnson Syndrome (SJS)"]:
            print(f"Using DIFFUSE LESION (Dermatitis/SJS) fallback for {disease}.")
            mask, count = diffuse_lesion_fallback(image)
        
        else: # Default: "Skin Lesion (Generic/Acne)"
            print("Using SKIN LESION (GENERIC) fallback.")
            mask, count = generic_lesion_fallback(image)
            
        return mask, count

# ---------------------------------------------------------------------
# --- DISEASE-SPECIFIC FALLBACK IMPLEMENTATIONS ---
# ---------------------------------------------------------------------

def generic_lesion_fallback(image):
    """Simple broad color thresholding for general skin lesions/acne patches (the original logic)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Broad red range
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Simple Morphological Operations
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.erode(mask, kernel, iterations = 1)
    mask = cv2.dilate(mask, kernel, iterations = 2)

    # Count is not reliable with broad segmentation
    return mask, 0


def nail_psoriasis_fallback(image):
    """
    Analyzes nail plate for texture/pitting changes. Uses Adaptive Thresholding.
    The COUNT metric here represents the number of detected pits/discolored spots.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance local contrast (good for finding pits/texture)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Adaptive Thresholding to find pits (darker areas relative to neighbors)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 25, 5)

    # Noise reduction
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pit_count = 0
    final_mask = np.zeros(mask.shape, dtype=np.uint8)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        # Filter by reasonable size for pits/spots
        if 10 < area < 1000: 
            cv2.drawContours(final_mask, [contour], -1, 255, thickness=cv2.FILLED)
            pit_count += 1

    return final_mask, pit_count


def diffuse_lesion_fallback(image):
    """
    Generalized segmentation for large, poorly defined patches (Dermatitis, SJS).
    Uses color and edge detection to define the large, inflamed area.
    """
    # 1. Broad Red/Inflammation Color Mask
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 30, 30])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 30, 30])
    upper_red2 = np.array([180, 255, 255])
    color_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    
    # 2. Edge Detection (to confirm boundaries)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # 3. Combine color and edges and close gaps
    kernel = np.ones((7,7),np.uint8)
    color_closed = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # 4. Find the largest contour (the main lesion patch)
    contours, _ = cv2.findContours(color_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Filter out very small noise patches
        if cv2.contourArea(largest_contour) > 1000: 
            final_mask = np.zeros_like(gray)
            cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            lesion_count = 1 # Treat the single large patch as one entity
        else:
            final_mask = np.zeros_like(gray)
            lesion_count = 0
    else:
        final_mask = np.zeros_like(gray)
        lesion_count = 0

    return final_mask, lesion_count

# ---------------------------------------------------------------------
# --- MEASUREMENT & MAIN FUNCTIONS (Unchanged) ---
# ---------------------------------------------------------------------

def calculate_lesion_area(mask):
    """Calculates the area of the segmented lesion in pixels."""
    # Count the white pixels (lesion) in the mask
    area = np.sum(mask > 0)
    return area

def main():
    # Load images
    img_past = cv2.imread(PAST_IMAGE_PATH)
    img_new = cv2.imread(NEW_IMAGE_PATH)

    if img_past is None or img_new is None:
        print(f"Error: Could not load one or both images. Check 'data/' folder and file names ({PAST_IMAGE_PATH}, {NEW_IMAGE_PATH}).")
        return

    # --- 2. SEGMENTATION (Defaults to 'Skin Lesion' for CLI execution) ---
    print("\nStarting Lesion Segmentation...")
    
    # Hardcode disease for testing in CLI environment, assuming default is generic lesion
    cli_disease = "Skin Lesion (Generic/Acne)" 
    
    mask_past, count_past = segment_lesion(img_past, disease=cli_disease) 
    mask_new, count_new = segment_lesion(img_new, disease=cli_disease) 

    # --- 3. MEASUREMENT ---
    area_past = calculate_lesion_area(mask_past)
    area_new = calculate_lesion_area(mask_new)

    print("\n--- QUANTITATIVE RESULTS ---")
    
    # Only display count if it's actually calculated (i.e., not 0 from the generic fallback)
    if count_past > 0 or count_new > 0:
        print(f"Past Lesion Count: {count_past}")
        print(f"New Lesion Count: {count_new}")
    
    print(f"Past Lesion Area: {area_past} pixels")
    print(f"New Lesion Area: {area_new} pixels")

    # --- 4. PROGRESS CALCULATION ---
    
    # Calculation based on AREA (General metric)
    if area_past > 0:
        percent_area_change = ((area_new - area_past) / area_past) * 100
        
        print("\n--- PROGRESS SUMMARY (Area) ---")
        if percent_area_change < 0:
            print(f"✅ PROGRESS: Lesion size decreased by: {abs(percent_area_change):.2f}%")
        elif percent_area_change > 0:
            print(f"❌ REGRESSION: Lesion size increased by: {percent_area_change:.2f}%")
        else:
            print("➡️ NO CHANGE: Area change is negligible.")
            
    # Calculation based on COUNT (Runs if the specialized fallback returns a count)
    if count_past > 0:
        percent_count_change = ((count_new - count_past) / count_past) * 100
        
        print("\n--- PROGRESS SUMMARY (Count) ---")
        if percent_count_change < 0:
            print(f"✅ PROGRESS: Lesion count decreased by: {abs(percent_count_change):.2f}%")
        elif percent_count_change > 0:
            print(f"❌ REGRESSION: Lesion count increased by: {percent_count_change:.2f}%")
        else:
            print("➡️ NO CHANGE: Count change is negligible.")
            
    elif area_past == 0 and count_past == 0:
        print("\nCould not find a lesion in the past image to establish a reliable baseline.")


    # --- 5. VISUALIZATION ---
    print("\nDisplaying Visualization...")
    img_past_masked = cv2.bitwise_and(img_past, img_past, mask=mask_past)
    img_new_masked = cv2.bitwise_and(img_new, img_new, mask=mask_new)
    
    combined_image = np.hstack((img_past, img_new, img_past_masked, img_new_masked))
    combined_image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(25, 12))
    plt.imshow(combined_image_rgb)
    plt.title("Past Image | New Image | Past Masked Lesion | New Masked Lesion", fontsize=16)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()