# AI-Based Skin Analysis & Care Recommendation Tool
deployed on-https://looksmaxxing.streamlit.app/
<img width="1440" height="900" alt="Screenshot 2025-07-18 at 22 37 43" src="https://github.com/user-attachments/assets/1aa3fc06-221c-47cc-95b1-287410f90a3f" />
<img width="1437" height="782" alt="Screenshot 2025-07-18 at 23 05 06" src="https://github.com/user-attachments/assets/d28311a9-ef3b-4031-9f01-89b7937d718b" />
<img width="855" height="634" alt="Screenshot 2025-07-18 at 23 47 29" src="https://github.com/user-attachments/assets/3648ffd4-c2f6-4a26-980a-6d25b18043cb" />



This web application leverages computer vision and machine learning to analyze facial images for common skin concerns and provides personalized skincare recommendations. Designed for accessibility and ease of use, it runs on Streamlit and requires no specialized hardware.

## Features
- **Acne Detection**: Identifies acne spots using image processing or a deep learning model (if available).
- **Oiliness Detection**: Analyzes skin shine and texture to estimate oiliness levels.
- **UV Exposure Analysis**: Assesses potential UV damage based on image characteristics.
- **Personalized Skincare Tips**: Provides actionable recommendations tailored to detected skin conditions.
- **Modular Design**: Each analysis (acne, oiliness, UV) is implemented as a separate module for easy extension.
- **User-Friendly Interface**: Simple upload-and-analyze workflow, suitable for all users.

## Technologies & Libraries Used
- **Streamlit**: For building the interactive web application interface. Chosen for its simplicity, rapid prototyping, and ability to create data-driven apps with minimal code.
- **OpenCV (opencv-python)**: Used for image processing, face detection, and feature extraction. Chosen for its robust computer vision capabilities and wide adoption in the industry.
- **NumPy**: Provides efficient numerical operations and image array manipulation. Chosen for its speed and ease of use in scientific computing.
- **Pillow (PIL)**: Handles image file loading, saving, and preprocessing. Chosen for its compatibility with various image formats and integration with other Python libraries.
- **TensorFlow** (optional): Used for deep learning-based acne detection if a model is provided. Chosen for its flexibility and support for deploying machine learning models.
- **Haar Cascade Classifier**: Utilized for robust face detection (using `haarcascade_frontalface_default.xml`). Chosen for its speed and effectiveness in real-time applications.

### Why These Tools?
- **Streamlit**: Enables quick development of interactive web apps without requiring frontend expertise.
- **OpenCV**: Industry-standard for image analysis, with extensive documentation and community support.
- **NumPy & Pillow**: Essential for efficient image data handling and manipulation.
- **TensorFlow**: Allows integration of advanced ML models for enhanced detection accuracy.
- **Haar Cascade**: Lightweight and effective for face detection, suitable for real-time analysis.

## Usage
1. Upload a selfie or face photo.
2. The app analyzes the image and displays results and recommendations.

### Local Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Cloud Deployment (Streamlit Community Cloud)
1. Push this repo to GitHub.
2. Go to https://streamlit.io/cloud and connect your repo.
3. Deploy! (No secrets or config needed for basic use.)

## Project Structure
```
LOOKSMAXXING-AI/
├── app.py                  # Streamlit web app
├── skin_analysis/
│   ├── __init__.py
│   ├── acne.py             # Acne detection logic
│   ├── oiliness.py         # Oiliness detection logic
│   ├── uv.py               # UV exposure analysis logic
│   └── ...                 # Other analysis modules
├── requirements.txt
├── README.md
├── LICENSE
├── haarcascade_frontalface_default.xml # Face detection model
└── (acne_model.h5)         # Optional: ML model for acne detection
```

## High-Level Design (HLD)
- The application is structured as a modular web app where each skin analysis feature (acne, oiliness, UV) is implemented as a separate module within the `skin_analysis/` directory.
- The main entry point (`app.py`) handles user interaction, image upload, and orchestrates the analysis pipeline.
- The design allows for easy extension by adding new analysis modules or swapping out detection algorithms.
- The system is stateless and processes each image independently, making it suitable for both local and cloud deployment.

## Low-Level Design (LLD)
- Each analysis module (e.g., `acne.py`, `oiliness.py`, `uv.py`) contains functions for processing images and returning results.
- The face detection logic uses OpenCV's Haar Cascade to locate faces before further analysis.
- If a deep learning model is present (`acne_model.h5`), TensorFlow is used to load and run predictions; otherwise, traditional image processing is used.
- Results from each module are aggregated and displayed in the Streamlit interface, along with personalized recommendations.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---
Questions? Open an issue or contact the maintainer.

## Credits

The Jupyter Notebook (.ipynb) files in this project are adapted from (https://github.com/Ankit-RV/Salon-Skin-Care-An-AI-Based-Skin-Analysis-Tool.git). Many thanks to the original author for their valuable work.






