# ==============================================================================
# SECTION 1 — Imports and configuration
# ==============================================================================
import json
import os
import time
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Tomato Disease Detector",
    page_icon="🍅",
    layout="centered",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = "model/tomato_disease_model.h5"
LABELS_PATH = "model/class_labels.json"
IMG_SIZE = (224, 224)


# ==============================================================================
# SECTION 2 — Model loading with caching
# ==============================================================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Please place tomato_disease_model.h5 inside the model/ folder.")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_labels():
    if not os.path.exists(LABELS_PATH):
        st.error(f"Labels file not found. Please place class_labels.json inside the model/ folder.")
        st.stop()
    with open(LABELS_PATH, "r") as f:
        return json.load(f)


model = load_model()
class_labels = load_labels()


# ==============================================================================
# SECTION 3 — Image preprocessing function
# ==============================================================================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array


# ==============================================================================
# SECTION 4 — Disease information dictionary
# ==============================================================================
DISEASE_INFO = {
    "Tomato Mosaic Virus": {
        "cause": "Tomato Mosaic Virus (ToMV) — highly contagious viral pathogen",
        "symptoms": "Mottled mosaic pattern on leaves, distorted growth, reduced fruit size",
        "treatment": "Remove and destroy infected plants. Disinfect tools. Use resistant varieties."
    },
    "Yellow Leaf Curl Virus": {
        "cause": "Tomato Yellow Leaf Curl Virus (TYLCV) — transmitted by whitefly Bemisia tabaci",
        "symptoms": "Severe leaf curling, yellowing, stunted growth, and greatly reduced yield",
        "treatment": "Control whitefly populations with insecticides. Use reflective mulches and resistant varieties."
    },
    "Bacterial Spot": {
        "cause": "Xanthomonas campestris pv. vesicatoria — bacterial pathogen",
        "symptoms": "Water-soaked dark lesions on leaves and fruit, surrounded by yellow halo",
        "treatment": "Apply copper-based bactericides. Avoid overhead irrigation. Remove infected debris."
    },
    "Early Blight": {
        "cause": "Alternaria solani — fungal pathogen",
        "symptoms": "Dark bull's-eye lesions with concentric rings, surrounded by yellow tissue on lower leaves",
        "treatment": "Apply fungicides (mancozeb or chlorothalonil). Improve air circulation. Rotate crops."
    },
    "Late Blight": {
        "cause": "Phytophthora infestans — highly destructive oomycete pathogen",
        "symptoms": "Water-soaked grey-green patches that rapidly turn brown. White fungal growth on leaf undersides.",
        "treatment": "Apply systemic fungicides immediately (metalaxyl). Remove infected material. Act fast — can destroy a field within days."
    },
    "Leaf Mold": {
        "cause": "Passalora fulva — fungal pathogen favoured by high humidity",
        "symptoms": "Yellow patches on upper leaf surface, olive-green or grey mold on underside",
        "treatment": "Improve ventilation. Reduce humidity. Apply fungicides. Remove infected leaves."
    },
    "Septoria Leaf Spot": {
        "cause": "Septoria lycopersici — fungal pathogen",
        "symptoms": "Numerous small circular spots with grey centres and dark brown margins, mainly on lower leaves",
        "treatment": "Remove infected lower leaves. Apply fungicides. Avoid wetting foliage when watering."
    },
    "Spider Mites (Two-Spotted Spider Mite)": {
        "cause": "Tetranychus urticae — arachnid pest, not a fungal or bacterial disease",
        "symptoms": "Fine stippling and bronzing of leaves. Fine webbing visible under heavy infestation.",
        "treatment": "Apply miticides or neem oil. Introduce predatory mites. Maintain humidity — mites thrive in dry conditions."
    },
    "Healthy": {
        "cause": "No disease or pest detected",
        "symptoms": "Leaf appears healthy with no visible lesions, discolouration, or deformation",
        "treatment": "Continue regular monitoring and good agricultural practices."
    }
}


# ==============================================================================
# SECTION 5 — App header
# ==============================================================================
st.markdown("<h1 style='text-align: center; margin-top: 0;'>Tomato Disease Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.15em; color: #4B5563;'>Upload a tomato leaf image to identify diseases and pest conditions</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: none; height: 2px; background-color: #2E7D32; margin: 15px 0 25px 0;' />", unsafe_allow_html=True)
st.info("This system classifies 8 tomato diseases and 1 pest condition. It is designed as a decision-support tool — always confirm with an agronomist for critical decisions.")


# ==============================================================================
# SECTION 6 — File uploader
# ==============================================================================
uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.markdown(
        """
        <div style="text-align: center; padding: 40px 20px; border: 2px dashed #A5D6A7; border-radius: 10px; background-color: #F1F8E9; margin-top: 15px;">
            <div style="font-size: 48px;">📷</div>
            <p style="font-size: 16px; color: #2E7D32; font-weight: bold; margin: 10px 0 5px 0;">Supported formats: JPG, JPEG, PNG</p>
            <p style="font-size: 14px; color: #555555; margin: 0;">For best results, use a clear close-up photo of a single leaf</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==============================================================================
# SECTION 7 — Prediction and results display
# ==============================================================================
else:
    col1, col2 = st.columns([1, 1])

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf Image")

    with col2:
        with st.spinner("Analysing leaf image..."):
            start_time = time.time()
            preprocessed_image = preprocess_image(image)
            predictions = model.predict(preprocessed_image, verbose=0)
            inference_time = time.time() - start_time

        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx] * 100)
        predicted_class = class_labels[str(predicted_idx)]

        st.markdown(f"## **{predicted_class}**")
        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

        if confidence >= 85:
            st.markdown(
                "<div style='background-color: #D4EDDA; color: #155724; padding: 10px 14px; border-radius: 6px; font-weight: bold; margin: 10px 0;'>"
                "High Confidence</div>",
                unsafe_allow_html=True
            )
        elif confidence >= 60:
            st.markdown(
                "<div style='background-color: #FFF3CD; color: #856404; padding: 10px 14px; border-radius: 6px; font-weight: bold; margin: 10px 0;'>"
                "Moderate Confidence — verify with an agronomist</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='background-color: #F8D7DA; color: #721C24; padding: 10px 14px; border-radius: 6px; font-weight: bold; margin: 10px 0;'>"
                "Low Confidence — image may be unclear or condition ambiguous</div>",
                unsafe_allow_html=True
            )

        st.caption(f"Analysis completed in {inference_time:.2f}s")

    # BELOW BOTH COLUMNS (full width)
    st.divider()
    st.subheader("Condition Details")

    info = DISEASE_INFO.get(predicted_class, {})
    col_cause, col_symptoms, col_treatment = st.columns(3)

    with col_cause:
        st.markdown("#### Cause")
        st.write(info.get("cause", "No information available."))

    with col_symptoms:
        st.markdown("#### Symptoms")
        st.write(info.get("symptoms", "No information available."))

    with col_treatment:
        st.markdown("#### Treatment")
        st.write(info.get("treatment", "No information available."))

    if predicted_class != "Healthy":
        st.warning(
            "⚠️ Disease Detected: Please consult an agricultural extension officer "
            "for confirmation and appropriate treatment. Do not apply treatments "
            "based solely on this system's prediction."
        )
    else:
        st.success("✅ No disease detected. Continue regular monitoring and good agricultural practices.")

    st.subheader("Prediction Confidence Across All Classes")
    chart_data = {
        "Class": [class_labels[str(i)] for i in range(len(class_labels))],
        "Probability": [float(predictions[0][i]) for i in range(len(class_labels))]
    }
    st.bar_chart(chart_data, x="Class", y="Probability")


# ==============================================================================
# SECTION 8 — Sidebar
# ==============================================================================
with st.sidebar:
    st.title("About This System")
    st.write(
        "This system was developed by Taiwo Eyitayo Omotoyosi as a final year Computer Science "
        "project at Oduduwa University, Ipetumodu."
    )

    st.divider()

    st.subheader("Model Details")
    st.markdown(
        """
        • **Architecture:** EfficientNet-B3  
        • **Pre-trained on:** ImageNet  
        • **Fine-tuned on:** PlantVillage Dataset  
        • **Test Accuracy:** 90.87%  
        • **Weighted F1-Score:** 90.93%  
        • **Macro-Average AUC:** 0.9958  
        • **Model File:** tomato_disease_model.h5  
        """
    )

    st.divider()

    st.subheader("Target Classes")
    st.markdown(
        """
        • Tomato Mosaic Virus  
        • Yellow Leaf Curl Virus  
        • Bacterial Spot  
        • Early Blight  
        • Late Blight  
        • Leaf Mold  
        • Septoria Leaf Spot  
        • Spider Mites (Two-Spotted Spider Mite)  
        • Healthy  
        """
    )

    st.divider()

    st.markdown(
        "<small><em>This tool is for educational and decision-support purposes only. "
        "Always consult a qualified agronomist before applying any treatment.</em></small>",
        unsafe_allow_html=True
    )


# ==============================================================================
# SECTION 9 — Footer
# ==============================================================================
st.divider()
st.markdown(
    "<p style='text-align: center; font-size: 0.85em; color: #6B7280;'>"
    "Developed by Taiwo Eyitayo | Computer Science | Oduduwa University, Ipetumodu | 2026"
    "</p>",
    unsafe_allow_html=True
)
