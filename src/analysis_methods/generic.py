import cv2
import numpy as np

def generic_lesion_fallback(image):
    """
    Enhanced Segmentation for Generic Skin Lesions/Acne Patches.
    Uses improved color thresholding and morphological operations to detect red lesions/acne.
    Includes contour detection for accurate lesion counting.

    RETURNS: (mask_image, lesion_count)
    """
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Enhanced red range for acne/lesions (broader and more sensitive)
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 40, 40])
    upper_red2 = np.array([180, 255, 255])

    # Combine the two red ranges
    mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    # Enhanced Morphological Operations for better lesion detection
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)  # Remove noise
    mask = cv2.dilate(mask, kernel, iterations=2)  # Connect nearby lesions
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)  # Fill gaps

    # Contour detection for lesion counting
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lesion_count = 0
    final_mask = np.zeros_like(mask)

    for contour in contours:
        area = cv2.contourArea(contour)
        # Filter by reasonable size for acne lesions (adjust based on image scale)
        if 50 < area < 10000:  # Typical acne lesion size range
            cv2.drawContours(final_mask, [contour], -1, 255, thickness=cv2.FILLED)
            lesion_count += 1

    return final_mask, lesion_count
