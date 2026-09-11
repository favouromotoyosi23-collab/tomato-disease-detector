# 🍅 Tomato Disease Detector

A clean, professional Streamlit web application that classifies uploaded tomato leaf images into nine disease and pest conditions using a fine-tuned EfficientNet-B3 deep learning model.

---

## 🚀 How to Run Locally

1. **Clone or navigate to the project directory:**
   ```bash
   cd tomato_disease_app
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Place the model file:**
   Copy `tomato_disease_model.h5` into the `model/` folder before running:
   ```
   model/
   ├── class_labels.json
   └── tomato_disease_model.h5
   ```

4. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

---

## 📦 How to Place the Model File

Copy `tomato_disease_model.h5` into the `model/` folder before running the application. The system expects the file at `model/tomato_disease_model.h5`. If the model file is not present, the app will cleanly display a notification prompting you to place `tomato_disease_model.h5` in the `model/` folder.

---

## ☁️ How to Deploy to Streamlit Community Cloud

1. Push the project folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"**, select the repository, and set the main file path to `app.py`.
4. Ensure tomato_disease_model.h5 is inside the model/ folder in the repo (file is ~46 MB — within GitHub's standard 100 MB limit, no Git LFS needed).
5. Click **Deploy** — the app will be live within 2 to 3 minutes.

---

## 🌿 Nine Target Classes

The system detects and classifies the following 8 tomato diseases and 1 pest condition:

1. **Tomato Mosaic Virus** (Viral pathogen)
2. **Yellow Leaf Curl Virus** (Viral pathogen transmitted by whitefly)
3. **Bacterial Spot** (Bacterial pathogen *Xanthomonas campestris pv. vesicatoria*)
4. **Early Blight** (Fungal pathogen *Alternaria solani*)
5. **Late Blight** (Oomycete pathogen *Phytophthora infestans*)
6. **Leaf Mold** (Fungal pathogen *Passalora fulva*)
7. **Septoria Leaf Spot** (Fungal pathogen *Septoria lycopersici*)
8. **Spider Mites (Two-Spotted Spider Mite)** (Arachnid pest *Tetranychus urticae*)
9. **Healthy** (No disease or pest detected)

---

## 📊 Model Performance Figures

The underlying deep learning classifier was developed and evaluated with the following specifications and metrics:

- **Architecture:** EfficientNet-B3
- **Pre-trained on:** ImageNet
- **Fine-tuned on:** PlantVillage Dataset
- **Test Accuracy:** 90.87%
- **Weighted F1-Score:** 90.93%
- **Macro-Average AUC:** 0.9958
- **Model File:** `tomato_disease_model.h5`

---

## 👨‍💻 Project Information

Developed by **Taiwo Eyitayo** | Computer Science | Oduduwa University, Ipetumodu | 2025  
*Disclaimer: This tool is for educational and decision-support purposes only. Always consult a qualified agronomist before applying any treatment.*
