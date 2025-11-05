"""
Spin Analysis Module for Disease Progress Tracker

This module provides analysis functions for specific disease types.
Includes scoliosis analysis for detecting spinal curvature in physical images.
"""

import cv2
import numpy as np
from scipy.optimize import curve_fit

def analyze_spin_features(image, disease_type="spin"):
    """
    Analyze spin-like features in images (placeholder for specific analysis).

    This could be used for analyzing rotational patterns, spiral features,
    or other specific morphological characteristics in skin lesions.

    Parameters:
    - image: Input image (BGR format)
    - disease_type: Type of disease being analyzed

    Returns:
    - mask: Binary mask of detected features
    - count: Number of detected features
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding to detect potential features
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours to count features
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = len(contours)

    return mask, count

def segment_lesion_spin(image, disease="spin"):
    """
    Segment lesions using spin-specific analysis.

    This is a placeholder function that can be extended for specific
    disease analysis requiring rotational or spiral feature detection.

    Parameters:
    - image: Input image
    - disease: Disease type (for future extension)

    Returns:
    - mask: Segmentation mask
    - count: Number of detected lesions/features
    """
    mask, count = analyze_spin_features(image, disease)
    return mask, count

def calculate_spin_area(mask):
    """
    Calculate the total area of spin features.

    Parameters:
    - mask: Binary mask

    Returns:
    - area: Total area in pixels
    """
    return np.sum(mask > 0)

def scoliosis_fallback(image):
    """
    Fallback analysis for scoliosis using edge detection and curvature calculation.
    Detects spinal curvature in physical images (e.g., posture photos) by finding the main spine contour
    and calculating a curvature metric (e.g., polynomial fit deviation).

    Parameters:
    - image: Input image (BGR format)

    Returns:
    - mask: Binary mask highlighting detected spine features
    - curvature_metric: A numeric value representing curvature severity (higher = more curved)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate edges to connect potential spine lines
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours to find the most likely spine (vertical, tall, central)
    filtered_contours = []
    img_height, img_width = gray.shape
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:  # Lower threshold for small images
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / w if w > 0 else 0
        if aspect_ratio < 1.2:  # Less strict for small images
            continue
        if w > h * 0.8:  # Allow wider for small images
            continue
        # Prefer central contours
        center_x = x + w / 2
        if center_x < img_width * 0.2 or center_x > img_width * 0.8:
            continue
        length = cv2.arcLength(contour, False)
        filtered_contours.append((contour, length, area))

    if filtered_contours:
        # Choose the one with longest arc length among filtered
        spine_contour, max_length, _ = max(filtered_contours, key=lambda x: x[1])
    else:
        # If no filtered contours, try the largest by area
        if contours:
            spine_contour = max(contours, key=cv2.contourArea)
        else:
            spine_contour = None

    if spine_contour is None or len(spine_contour) < 10:
        # No significant contour found, return empty mask and zero curvature
        mask = np.zeros_like(gray)
        return mask, 0

    # Fit a polynomial to the contour points to approximate curvature
    points = spine_contour.squeeze()
    if points.ndim == 1:
        points = points.reshape(-1, 1, 2).squeeze()
    if len(points) < 3:
        mask = np.zeros_like(gray)
        return mask, 0

    # Sort points by y-coordinate (assuming vertical spine)
    points = points[np.argsort(points[:, 1])]

    # Fit a quadratic polynomial: x = a*y^2 + b*y + c
    y_vals = points[:, 1].astype(float)
    x_vals = points[:, 0].astype(float)

    try:
        # Polynomial fit (degree 2 for curvature)
        coeffs = np.polyfit(y_vals, x_vals, 2)
        poly = np.poly1d(coeffs)

        # Calculate curvature metric as the coefficient of y^2 (indicates how curved it is)
        curvature_metric = abs(coeffs[0]) * 1000  # Scale for readability

        # Create mask by drawing the fitted curve
        mask = np.zeros_like(gray)
        y_range = np.linspace(np.min(y_vals), np.max(y_vals), 100)
        x_fitted = poly(y_range)
        pts = np.array([np.column_stack((x_fitted, y_range))], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)

    except np.linalg.LinAlgError:
        # Fit failed, use contour area as fallback metric
        curvature_metric = cv2.contourArea(spine_contour)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [spine_contour], -1, 255, thickness=cv2.FILLED)

    return mask, int(curvature_metric)

# Example usage and testing
if __name__ == "__main__":
    # Test with a sample image
    import os

    test_image_path = "data/dermatitis_past/eczema_dermatitis_skin_16.jpg"
    if os.path.exists(test_image_path):
        test_img = cv2.imread(test_image_path)
        if test_img is not None:
            mask, count = segment_lesion_spin(test_img)
            area = calculate_spin_area(mask)
            print(f"Detected {count} features with total area of {area} pixels")
        else:
            print("Could not load test image")
    else:
        print("Test image not found")
