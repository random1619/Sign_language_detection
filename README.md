# 🖐️ Sign Language Detection

A deep learning-based project for detecting and translating sign language gestures into human-readable text. This tool aims to improve communication accessibility for individuals using sign language.

## 🚀 Features

- Real-time sign language recognition using webcam
- Supports American Sign Language (ASL) alphabet (A-Z)
- CNN-based gesture classification
- Preprocessing and hand landmark extraction using MediaPipe
- Easy-to-use UI and modular code structure

## 📸 Demo

![Demo](https://github.com/yourusername/sign-language-detection/assets/demo.gif)

## 🛠️ Tech Stack

- **Python 3.x**
- **OpenCV** – for image capture and processing
- **MediaPipe** – for hand tracking and landmark extraction
- **TensorFlow / Keras** – for building and training the CNN model
- **NumPy / Pandas** – for data manipulation
- **Matplotlib / Seaborn** – for visualizations (optional)
- **Streamlit** or **Tkinter** – for a simple frontend (optional)

## 📁 Project Structure
sign-language-detection/
├── data/                # Dataset directory
├── model/               # Trained models and checkpoints
│   └── keypoint_classifier/
│       ├── keypoint.csv
│       └── keypoint_classifier_label.csv
├── utils/               # Helper functions
├── notebooks/           # Jupyter notebooks for training
│   ├── keypoint_classification.ipynb
├── app.py               # Main application file
├── interface.py         # Streamlit interface
├── requirements.txt     # Dependencies
└── README.md            # Project documentation


## 🧠 Model Training

The model is a CNN trained on a dataset of labeled ASL alphabet images.

### Architecture

- Input layer: 64x64 grayscale images
- Convolutional + ReLU + MaxPooling (2-3 layers)
- Dense layers with Dropout
- Output layer with 26 neurons (A-Z)

### Dataset

We used the [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) from Kaggle which contains over 87,000 labeled images.

## ⚙️ Installation

1. **Clone the repository:**
git clone https://github.com/yourusername/sign-language-detection.git
cd sign-language-detection 

2. ** Set up a virtual environment and install dependencies:**
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt

4. For running the app
   python app.py

