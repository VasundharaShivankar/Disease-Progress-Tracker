import cv2
import numpy as np

def nail_psoriasis_fallback(image):
    """
    Segmentation for Nail Psoriasis.
    Analyzes nail plate for texture/pitting changes using Adaptive Thresholding.
    
    RETURNS: (mask_image, detected_pit_count)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance local contrast (CLAHE is good for finding pits/texture on nails)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Adaptive Thresholding to find pits (darker areas relative to neighbors)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 25, 5) # Invert for dark spots

    # Noise reduction
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find and count contours (representing pits/spots)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pit_count = 0
    final_mask = np.zeros(mask.shape, dtype=np.uint8)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        # Filter by reasonable size for typical nail pits/spots (adjust these values based on images)
        if 10 < area < 1000: 
            cv2.drawContours(final_mask, [contour], -1, 255, thickness=cv2.FILLED)
            pit_count += 1

    return final_mask, pit_count