# 🦴 Bone Fracture Detection

A deep learning web application that uses a **Convolutional Neural Network (CNN)** to detect bone fractures from X-ray images.
**Dataset:**[ ahmedashrafahmed/bone-fracture](https://www.kaggle.com/datasets/ahmedashrafahmed/bone-fracture)

## 🚀 Overview

This project takes an uploaded X-ray image, preprocesses it to the required format, and uses a trained CNN model to predict whether the image shows a **fracture** or **no fracture**.

The model is integrated into a **Flask web application**, allowing users to upload X-ray images directly through a browser.

## 🧠 Technologies Used

* Python
* TensorFlow / Keras
* CNN
* Flask
* NumPy
* Pillow
* HTML & CSS

## 🔄 Project Workflow

```text
X-Ray Image
     ↓
Image Upload
     ↓
Image Preprocessing
     ↓
Resize to 256 × 256
     ↓
CNN Model
     ↓
Prediction
     ↓
Fracture / No Fracture
```

## 📁 Project Structure

```text
Bone-Fracture-Detection/
│
├── app.py
├── bone-fracture.keras
├── requirements.txt
├── templates/
│   └── index.html
└── README.md
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repository.git
cd Bone-Fracture-Detection
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Flask server:

```bash
python app.py
```

Open the local URL shown in the terminal, usually:

```text
http://127.0.0.1:5000/
```

Upload an X-ray image and the model will generate a prediction.

## 📊 Model

The project uses a trained CNN model stored in:

```text
bone-fracture.keras
```

Input images are resized to **256 × 256 pixels** before being passed to the model.

## ⚠️ Disclaimer

This project is intended for **educational and demonstration purposes only**. It is not a medical diagnostic tool and should not be used as a replacement for professional medical evaluation.

## 👨‍💻 Author

**Asad Ajaz**
