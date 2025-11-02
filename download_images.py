import os
import requests
from bs4 import BeautifulSoup
import urllib.request
import time
import random

# IMPORTANT LEGAL AND ETHICAL WARNINGS:
# ======================================================================
# 1. MEDICAL IMAGES: Downloading medical images from the internet without
#    proper permissions violates patient privacy laws (HIPAA, GDPR, etc.)
#    and may contain copyrighted material.
#
# 2. COPYRIGHT: Most images found online are copyrighted. Using them for
#    training AI models may violate copyright laws.
#
# 3. QUALITY: Random internet images are not suitable for medical training
#    due to lack of proper diagnosis, quality control, and annotation.
#
# 4. RECOMMENDATION: Use public medical datasets instead:
#    - ISIC (International Skin Imaging Collaboration): https://isic-archive.com/
#    - DermNet: https://dermnetnz.org/
#    - MedMNIST: https://medmnist.com/
#    - NIH Chest X-ray dataset: https://nihcc.app.box.com/v/ChestXray-NIHCC
#
# THIS SCRIPT IS FOR EDUCATIONAL PURPOSES ONLY.
# DO NOT USE DOWNLOADED IMAGES FOR MEDICAL DIAGNOSIS OR TRAINING.
# ======================================================================

def download_google_images(search_term, num_images=50, output_dir="data"):
    """
    Download images from Google Images search.
    THIS IS FOR EDUCATIONAL PURPOSES ONLY - NOT FOR MEDICAL USE.
    """
    # Create output directory
    disease_dir = os.path.join(output_dir, search_term.lower().replace(" ", "_"))
    os.makedirs(disease_dir, exist_ok=True)

    # Try multiple search approaches since Google blocks simple scraping
    search_urls = [
        f"https://www.google.com/search?q={search_term.replace(' ', '+')}&tbm=isch",
        f"https://www.bing.com/images/search?q={search_term.replace(' ', '+')}",
        f"https://www.dogpile.com/search/images?q={search_term.replace(' ', '+')}"
    ]

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    img_urls = []

    for search_url in search_urls:
        try:
            print(f"Trying {search_url.split('/')[2]}...")
            response = requests.get(search_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Try different selectors for different search engines
            if 'google.com' in search_url:
                # Google Images
                img_tags = soup.find_all('img', {'src': True})
            elif 'bing.com' in search_url:
                # Bing Images
                img_tags = soup.find_all('img', {'src': True})
            else:
                # Dogpile
                img_tags = soup.find_all('img', {'src': True})

            for img in img_tags:
                src = img.get('src') or img.get('data-src')
                if src and src.startswith('http') and not any(skip in src.lower() for skip in ['favicon', 'logo', 'icon']):
                    img_urls.append(src)
                    if len(img_urls) >= num_images * 3:  # Get more to account for failures
                        break

            if len(img_urls) >= num_images:
                break

            time.sleep(1)  # Be respectful

        except Exception as e:
            print(f"Error with {search_url.split('/')[2]}: {e}")
            continue

    print(f"Found {len(img_urls)} potential images for {search_term}")

    # Download images
    downloaded = 0
    for i, url in enumerate(img_urls[:num_images]):
        try:
            filename = f"{search_term.lower().replace(' ', '_')}_{i+1}.jpg"
            filepath = os.path.join(disease_dir, filename)

            # Add headers for download
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                with open(filepath, 'wb') as f:
                    f.write(response.read())

            downloaded += 1

            # Be respectful to servers
            time.sleep(random.uniform(0.5, 1.5))

            if downloaded % 5 == 0:
                print(f"Downloaded {downloaded} images for {search_term}")

        except Exception as e:
            print(f"Failed to download image {i+1}: {e}")
            continue

    print(f"Successfully downloaded {downloaded} images for {search_term}")

def main():
    """
    Download sample images for different disease types.
    REMEMBER: This is for demonstration only. Use proper medical datasets for real training.
    """

    print("=" * 80)
    print("MEDICAL IMAGE DOWNLOAD WARNING")
    print("=" * 80)
    print("This script downloads images from Google for demonstration purposes.")
    print("DO NOT use these images for medical training or diagnosis.")
    print("Use proper medical datasets from academic sources instead.")
    print("=" * 80)

    # Disease search terms (generic, non-medical terms for demo)
    diseases = {
        "acne_past": "acne skin condition",
        "dermatitis_past": "eczema dermatitis skin",
        "nail_psoriasis_past": "nail psoriasis condition",
        "sjs_past": "stevens johnson syndrome skin"
    }

    for disease_folder, search_term in diseases.items():
        print(f"\nDownloading images for {disease_folder}...")
        download_google_images(search_term, num_images=20)  # Small number for demo

        # Respectful delay between searches
        time.sleep(2)

    print("\nDownload complete.")
    print("REMINDER: These images are NOT suitable for medical training.")
    print("Please use proper medical datasets from academic sources.")

if __name__ == "__main__":
    main()
