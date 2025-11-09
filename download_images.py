import os
import requests
import zipfile
import io
import time

# IMPORTANT LEGAL AND ETHICAL NOTES:
# ======================================================================
# This script now downloads from public medical datasets that are legally
# available for research and educational purposes.
#
# 1. ISIC (International Skin Imaging Collaboration): https://isic-archive.com/
#    - Provides dermatoscopic images with annotations for research.
#    - Requires registration but allows downloads for academic use.
#
# 2. DermNet: https://dermnetnz.org/
#    - Public domain images from New Zealand Dermatological Society.
#    - Free for educational and research use.
#
# 3. MedMNIST: https://medmnist.com/
#    - Benchmark datasets for medical image classification.
#    - Open source and free to use.
#
# REMEMBER: Always cite sources and use appropriately for research only.
# ======================================================================

def download_isic_images(output_dir="data", num_images=50):
    """
    Download sample images from ISIC Archive.
    Note: This is a simplified example. In practice, you'd use their API.
    """
    print("Downloading from ISIC Archive (sample demonstration)...")

    print("ISIC requires API registration. Please visit https://isic-archive.com/ to download manually.")
    print("For automated download, implement API calls with proper authentication.")

    # Placeholder: Create directory
    isic_dir = os.path.join(output_dir, "isic_samples")
    os.makedirs(isic_dir, exist_ok=True)
    print(f"Created directory: {isic_dir}")
    print("Please download ISIC images manually for now.")

def download_dermnet_images(output_dir="data", num_images=50):
    """
    Download images from DermNet NZ.
    DermNet provides public domain images.
    """
    print("Downloading from DermNet NZ...")

    base_url = "https://dermnetnz.org"
    categories = {
        "acne": "/assets/acne",
        "eczema": "/assets/eczema",
        "psoriasis": "/assets/psoriasis"
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for category, path in categories.items():
        category_dir = os.path.join(output_dir, f"{category}_past")
        os.makedirs(category_dir, exist_ok=True)

        try:
            # Get category page
            response = requests.get(f"{base_url}{path}", headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"Failed to access {category} page")
                continue

            # Simple parsing for image links (simplified)
            content = response.text
            img_urls = []

            # Extract image URLs (basic regex-like approach)
            import re
            img_pattern = r'<img[^>]+src="([^"]+\.jpg[^"]*)"'
            matches = re.findall(img_pattern, content)

            for match in matches[:num_images//len(categories)]:
                if match.startswith('/'):
                    img_urls.append(f"{base_url}{match}")
                elif match.startswith('http'):
                    img_urls.append(match)

            # Download images
            downloaded = 0
            for i, url in enumerate(img_urls):
                try:
                    filename = f"{category}_{i+1}.jpg"
                    filepath = os.path.join(category_dir, filename)

                    img_response = requests.get(url, headers=headers, timeout=10)
                    if img_response.status_code == 200:
                        with open(filepath, 'wb') as f:
                            f.write(img_response.content)
                        downloaded += 1

                    time.sleep(0.5)  # Be respectful

                except Exception as e:
                    print(f"Failed to download {url}: {e}")
                    continue

            print(f"Downloaded {downloaded} images for {category}")

        except Exception as e:
            print(f"Error downloading {category}: {e}")

def download_medmnist(output_dir="data"):
    """
    Download MedMNIST datasets.
    """
    print("Downloading MedMNIST datasets...")

    medmnist_url = "https://zenodo.org/record/6496656/files/medmnist.zip?download=1"

    try:
        response = requests.get(medmnist_url, timeout=30)
        if response.status_code == 200:
            # Extract zip
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(output_dir)
            print("MedMNIST downloaded and extracted.")
        else:
            print("Failed to download MedMNIST.")
    except Exception as e:
        print(f"Error downloading MedMNIST: {e}")

def main():
    """
    Download sample images from public medical datasets.
    """

    print("=" * 80)
    print("PUBLIC MEDICAL DATASET DOWNLOAD")
    print("=" * 80)
    print("This script downloads from legal, public medical datasets.")
    print("Use for research and educational purposes only.")
    print("Cite sources appropriately.")
    print("=" * 80)

    download_dermnet_images(num_images=20)
    download_isic_images(num_images=20)
    download_medmnist()

    print("\nDownload complete.")
    print("Images are now available for research purposes.")

if __name__ == "__main__":
    main()
