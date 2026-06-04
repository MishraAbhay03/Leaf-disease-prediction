# 🌿 Leaf Disease Detection using CNN

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> A deep learning system that detects **10+ plant diseases** from leaf images using a CNN trained on the PlantVillage dataset — achieving **86.78% accuracy** with transfer learning.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Accuracy | **86.78%** |
| Diseases Detected | **10+** |
| Dataset | PlantVillage |
| Architecture | CNN + Transfer Learning |

---

## 🧠 Model Architecture

```
 Input Image (256x256x3)
       │
       ▼
 Pretrained Base (Transfer Learning)
       │
       ▼
 Custom CNN Layers
       │
       ▼
 Data Augmentation
       │
       ▼
 Dense + Dropout Layers
       │
       ▼
 Softmax Output (10+ Disease Classes)
```

---

## 🦠 Diseases Detected

- Apple Scab, Apple Black Rot, Cedar Apple Rust
- Corn Common Rust, Corn Northern Leaf Blight
- Grape Black Rot, Grape Leaf Blight
- Potato Early Blight, Potato Late Blight
- Tomato Bacterial Spot, and more...

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Deep Learning | TensorFlow / Keras |
| Image Processing | OpenCV |
| Data Augmentation | Keras ImageDataGenerator |
| Web App | Streamlit |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
├── Leaf_Diseases_Prediction.ipynb   # Full training notebook
├── app.py                           # Streamlit web app
├── plant_disease_model.h5           # Trained model weights
├── requirements.txt                 # Dependencies
└── README.md
```

---

## 🚀 Getting Started

```bash
# Clone
git clone https://github.com/MishraAbhay03/Leaf-disease-prediction.git
cd Leaf-disease-prediction

# Install
pip install -r requirements.txt

# Run the web app
streamlit run app.py
```

Upload a leaf image and the model will predict whether it's healthy or infected, along with the disease name.

---

## 🔬 Key Techniques

- **Transfer Learning** — pretrained weights for faster convergence
- **Data Augmentation** — rotation, zoom, flip to prevent overfitting
- **Dropout Regularization** — reduces overfitting on limited labeled data
- **Softmax Classification** — multi-class output for 10+ disease types

---

## 👤 Author

**Abhaykumar Mishra**  
M.Sc. Data Science & AI | Mumbai  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/YOUR_LINKEDIN) [![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/MishraAbhay03)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
