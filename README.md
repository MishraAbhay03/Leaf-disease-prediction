# 🌿 Leaf Disease Detection (CNN)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Kaggle](https://img.shields.io/badge/Dataset-PlantVillage-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

> CNN-based plant disease classifier achieving **86.78% accuracy** across 10+ disease categories — trained on the PlantVillage dataset using TensorFlow and transfer learning.

---

## 📌 Overview

This project uses a Convolutional Neural Network (CNN) to detect and classify plant leaf diseases from images. Accurate early detection of plant diseases can dramatically reduce crop losses and improve agricultural yield.

---

## ✨ Key Features

- ✅ **86.78% accuracy** on 10+ plant disease categories
- ✅ Transfer learning with pretrained CNN architectures
- ✅ Data augmentation to handle limited labeled data
- ✅ Interactive **Streamlit** web app for real-time prediction
- ✅ OpenCV preprocessing pipeline
- ✅ Trained on **PlantVillage** dataset

---

## 🌱 Disease Categories Detected

| # | Disease |
|---|---|
| 1 | Apple Scab |
| 2 | Black Rot |
| 3 | Cedar Apple Rust |
| 4 | Corn Grey Leaf Spot |
| 5 | Grape Black Rot |
| 6 | Potato Early Blight |
| 7 | Potato Late Blight |
| 8 | Tomato Bacterial Spot |
| 9 | Tomato Early Blight |
| 10 | Healthy (control) |

---

## 📁 Project Structure

```
Leaf-disease-prediction/
├── Leaf_Diseases_Prediction.ipynb   # Model training notebook
├── app.py                           # Streamlit web app
├── plant_disease_model.h5           # Trained model weights
├── requirements.txt                 # Dependencies
└── README.md
```

---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/MishraAbhay03/Leaf-disease-prediction.git
cd Leaf-disease-prediction
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

Upload a leaf image and get instant disease classification!

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | **86.78%** |
| Dataset | PlantVillage |
| Classes | 10+ |
| Framework | TensorFlow/Keras |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | TensorFlow, Keras |
| Image Processing | OpenCV, PIL |
| Web App | Streamlit |
| Language | Python 3.9+ |

---

## 👤 Author

**Abhaykumar Mishra** — [GitHub](https://github.com/MishraAbhay03) · [LinkedIn](https://linkedin.com/in/YOUR_LINKEDIN)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
