import cv2
import numpy as np

def generic_lesion_fallback(image):
    """
    Segmentation for Generic Skin Lesions/Acne Patches.
    Uses broad red color thresholding in HSV color space.
    
    RETURNS: (mask_image, lesion_count)
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Broad red range (0-10 and 170-180 in Hue)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Combine the two red ranges
    mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Simple Morphological Operations (Cleanup)
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.erode(mask, kernel, iterations = 1)
    mask = cv2.dilate(mask, kernel, iterations = 2)

    # Lesion count is not reliable with broad segmentation
    return mask, 0