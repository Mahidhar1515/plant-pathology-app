# 🌿 Plant Pathology Detection App

This project is a web-based Plant Pathology Detection system built with **Flask**. It uses a pre-trained **TensorFlow Lite (TFLite)** model to classify plant diseases from uploaded images.

## 🚀 Features

- Upload plant images to detect diseases
- Uses lightweight `.tflite` models for fast inference
- Clean UI built with HTML/CSS (Bootstrap-compatible)
- REST API backend using Flask
- Support for out-of-distribution (OOD) detection (optional)

## 🧠 Model

- Main model: `plant_disease_model.tflite`
- Trained using TensorFlow/Keras and converted from `.h5` format
- Additional models and data files stored in the `/models` directory

## 📁 Project Structure

ppi/
│
├── app.py # Main Flask application
├── convert_model.py # Script to convert .h5 to .tflite
├── train.py # Model training script
├── requirements.txt # Python dependencies
│
├── models/
│ ├── plant_disease_model.tflite
│ ├── best_crop_disease_model.h5
│ ├── centroids_and_threshold.pkl
│ └── crop_disease_model.h5
│
├── static/
│ └── images/
│ └── bgd.jpg # Background image
│
├── templates/
│ └── index.html # Web UI template


## 🧪 Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/plant-pathology-app.git
   cd plant-pathology-app
2. Create a virtual environment and activate it:
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
3. Install dependencies:
     pip install -r requirements.txt
4. Run the Flask app:
   python app.py

