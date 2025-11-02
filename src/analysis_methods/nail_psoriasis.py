import cv2
import numpy as np

def nail_psoriasis_fallback(image):
    """
    Enhanced Segmentation for Nail Psoriasis.
    Analyzes nail plate for multiple features: pitting, discoloration, and separation (onycholysis).
    Combines texture-based pit detection with color-based discoloration detection.

    RETURNS: (mask_image, detected_feature_count)
    """
    # Convert to HSV for color-based detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 1. Detect discoloration (yellow/red hues typical in psoriasis)
    # Yellow range (for discoloration)
    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Red range (for inflammation)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine color masks
    color_mask = cv2.bitwise_or(yellow_mask, red_mask)

    # Morphological operations to refine color mask
    kernel = np.ones((5,5), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    color_mask = cv2.erode(color_mask, kernel, iterations=1)

    # 2. Detect pitting/texture changes
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance local contrast (CLAHE for pits)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Adaptive Thresholding for pits (darker spots)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 2)  # Adjusted params for sensitivity

    # Noise reduction for pit mask
    pit_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # 3. Combine color and pit masks
    combined_mask = cv2.bitwise_or(color_mask, pit_mask)

    # Further refinement: remove small noise
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 4. Find contours for feature counting
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    feature_count = 0
    final_mask = np.zeros(combined_mask.shape, dtype=np.uint8)

    for contour in contours:
        area = cv2.contourArea(contour)
        # Filter contours by size (pits/discoloration patches)
        if 20 < area < 5000:  # Adjusted range for nail features
            cv2.drawContours(final_mask, [contour], -1, 255, thickness=cv2.FILLED)
            feature_count += 1

    return final_mask, feature_count
