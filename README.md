# Disease Progress Tracker

## Author
This project is authored by Vasundhara Shivankar.

## Project Overview
The Disease Progress Tracker is a web application designed to analyze and track the progression of skin and spinal diseases in Children. It leverages deep learning models and advanced image processing techniques to provide insights into skin conditions (such as acne, dermatitis, hyperpigmentation, psoriasis, and nail psoriasis) and scoliosis through detailed image analysis and classification.

The application offers functionalities such as disease classification from skin images, comparison between past and recent lesion images to measure progress, and scoliosis severity analysis with supplement recommendations.

## Features
- Skin disease classification using deep learning (VGG16-based model).
- Lesion segmentation and area calculation to monitor improvement over time.
- Scoliosis image analysis to detect spinal abnormalities and suggest supplements.
- Interactive web interface for uploading images and viewing reports.
- REST API endpoints to predict skin diseases and serve analysis results.

## Technology Stack
- Backend:
  - Python, Flask Web Framework
  - TensorFlow and Keras for deep learning model inference
  - OpenCV and PIL for image processing
  - Matplotlib for visualization
  - Flask-CORS for cross-origin resource sharing
- Frontend:
  - ReactJS (served separately; CORS configured for localhost:3000)
- Database: No database integrated in current version
- Other Dependencies:
  - Various Python libraries (numpy, matplotlib, Pillow, flask, tensorflow, etc.)
  - Node.js dependencies managed by package.json

## Installation and Setup

### Prerequisites
- Python 3.7 or higher
- Node.js and npm (for frontend, if needed)
- Virtual environment tool (optional but recommended)

### Backend Setup
1. Clone the repository to your local machine.
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv\Scripts\activate
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure the pre-trained skin disease classification model `skinmodel_vgg16.h5` is present in the project root.
5. Make sure the `uploads/` folder exists in the root directory (it will be created automatically if not).
6. Run the Flask application:
   ```bash
   python app.py
   ```

### Frontend Setup
- The frontend is expected to run on `localhost:3000`.
- Navigate to the frontend directory `client/` and install dependencies:
  ```bash
  cd client
  npm install
  npm start
  ```

## Usage
- Open your browser and go to the frontend URL (usually `http://localhost:3000`).
- Use the provided interface to:
  - Upload skin images for classification.
  - Upload pairs of past and new lesion images for progress tracking.
  - Upload spine images for scoliosis analysis.
- View detailed reports, visualizations, and supplement recommendations based on analysis.

## Project Structure
- `app.py`: Main Flask backend application handling routes and analysis.
- `skinmodel_vgg16.h5`: Pre-trained skin disease classification model.
- `uploads/`: Folder to store uploaded images and generated visualizations.
- `src/`: Contains core analysis logic modules (skin and scoliosis analysis).
- `client/`: Frontend React app sources and assets.

## License
This project is licensed under the ISC License.

---

