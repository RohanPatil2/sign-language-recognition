

### 📌 **Modern README for Sign Language Recognition**

# 🖐️ Sign Language Recognition
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20Transformers-green)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

🚀 This repository contains code for **Sign Language Recognition**, developed as part of our **Final Year Project**.  
It uses **Mediapipe Hands**, **Blazepose**, and **Deep Learning Models (Transformers, LSTMs, XGBoost)** for accurate gesture recognition.

---

## 📖 Table of Contents
- [📂 Dataset](#-dataset)
- [⚙️ Installation](#️-installation)
- [💡 Usage](#-usage)
- [📌 Training & Evaluation](#-training--evaluation)
- [📊 Model Inference](#-model-inference)
- [👤 Authors](#-authors)
- [📜 License](#-license)

---

## 📂 Dataset
The **INCLUDE 50 Dataset** is used for training and evaluation.  
📥 **[Download Here](https://zenodo.org/records/4010759)**  

---

## ⚙️ **Installation**

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/sign-language-recognition.git
cd sign-language-recognition
```

### **2️⃣ Create a Virtual Environment (Recommended)**
```bash
python -m venv signlang_env
source signlang_env/bin/activate  # For Linux/Mac
signlang_env\Scripts\activate     # For Windows
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 💡 **Usage**

### **🔹 1. Generate Keypoints**
Extract **Mediapipe Hands & Blazepose keypoints** from videos and save them.
```bash
python generate_keypoints.py --include_dir <path_to_dataset> --save_dir <path_to_save_keypoints> --dataset <include/include50>
```

---

## 📌 **Training & Evaluation**
### **🔹 2. Train a Model**
Train a **Transformer / LSTM / XGBoost** model on the dataset.
```bash
python runner.py --dataset <include/include50> --use_augs --model transformer --data_dir <path_to_keypoints>
```

### **🔹 3. Resume Training or Evaluate a Pretrained Model**
Use a **pretrained model** for either evaluation or continuing training.
```bash
python runner.py --dataset <include/include50> --use_augs --model transformer --data_dir <path_to_keypoints> --use_pretrained <evaluate/resume_training>
```

---

## 📊 **Model Inference**
### **🔹 4. Get Predictions from a Pretrained Model**
Perform **sign language predictions** on new videos.
```bash
python evaluate.py --data_dir <path_to_videos>
```

---

## 👤 **Authors**
### **🎓 Developed By**
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/RohanPatil2">
        <img src="https://avatars.githubusercontent.com/u/12345678?v=4" width="100px;" alt="Rohan Patil"/>
        <br><b>Rohan Patil</b>
      </a>
      <br>
      <a href="https://github.com/RohanPatil2"><img src="https://img.shields.io/badge/GitHub-%40RohanPatil2-black.svg"></a>
    </td>
    <td align="center">
      <a href="https://github.com/sherurox">
        <img src="https://avatars.githubusercontent.com/u/87654321?v=4" width="100px;" alt="Shreyas Khandale"/>
        <br><b>Shreyas Khandale</b>
      </a>
      <br>
      <a href="https://github.com/sherurox"><img src="https://img.shields.io/badge/GitHub-%40sherurox-black.svg"></a>
    </td>
  </tr>
</table>

---

## 📜 **License**
This project is licensed under the **MIT License**.

---

## 🌟 **Support & Contributions**
If you find this project helpful, please **⭐️ star** this repository!  
Feel free to **fork** it and contribute via **Pull Requests**! 🚀

---
```

-
