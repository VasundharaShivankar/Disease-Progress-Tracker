import cv2
import numpy as np

def diffuse_lesion_fallback(image):
    """
    Generalized segmentation for large, poorly defined patches (Dermatitis, SJS).
    Focuses on identifying large, inflamed areas using color and morphology.
    
    RETURNS: (mask_image, single_patch_count)
    """
    # 1. Broad Red/Inflammation Color Mask
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Slightly wider, less saturated red range for diffuse inflammation
    lower_red1 = np.array([0, 30, 30])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 30, 30])
    upper_red2 = np.array([180, 255, 255])
    color_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    
    # 2. Close gaps in the inflamed area (Morphological Closing)
    kernel = np.ones((7,7),np.uint8)
    color_closed = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # 3. Find the largest contour (assuming the progress tracker tracks the main inflamed area)
    contours, _ = cv2.findContours(color_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Filter out very small noise patches
        if cv2.contourArea(largest_contour) > 1000: 
            final_mask = np.zeros_like(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            lesion_count = 1 # Treat the single large patch as one entity
        else:
            final_mask = np.zeros_like(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            lesion_count = 0
    else:
        final_mask = np.zeros_like(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        lesion_count = 0

    return final_mask, lesion_count