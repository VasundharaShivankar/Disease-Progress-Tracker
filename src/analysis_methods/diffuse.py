import cv2
import numpy as np

def diffuse_lesion_fallback(image):
    """
    Enhanced segmentation for large, poorly defined patches (Dermatitis, Eczema, SJS).
    Focuses on identifying inflamed areas using multiple color channels and advanced morphology.
    Includes better handling of diffuse inflammation patterns.

    RETURNS: (mask_image, patch_count)
    """
    # Convert to HSV and LAB for comprehensive color analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # 1. Enhanced Red/Inflammation Detection in HSV
    lower_red1 = np.array([0, 25, 25])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([160, 25, 25])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    # 2. Pink/Inflamed Detection in LAB (complementary to red)
    lower_pink_lab = np.array([140, 130, 130])
    upper_pink_lab = np.array([180, 160, 160])
    pink_mask = cv2.inRange(lab, lower_pink_lab, upper_pink_lab)

    # 3. Combine color masks
    color_mask = cv2.bitwise_or(red_mask, pink_mask)

    # 4. Advanced Morphological Operations
    kernel_small = np.ones((3,3), np.uint8)
    kernel_large = np.ones((7,7), np.uint8)

    # Remove small noise
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    # Close gaps in inflamed areas
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    # Dilate to connect nearby inflamed regions
    color_mask = cv2.dilate(color_mask, kernel_small, iterations=1)

    # 5. Contour Analysis for Multiple Patches
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    patch_count = 0
    final_mask = np.zeros_like(color_mask)

    for contour in contours:
        area = cv2.contourArea(contour)
        # Filter by size for diffuse patches (larger than acne but can be multiple)
        if area > 2000:  # Minimum area for significant diffuse patches
            cv2.drawContours(final_mask, [contour], -1, 255, thickness=cv2.FILLED)
            patch_count += 1

    return final_mask, patch_count
