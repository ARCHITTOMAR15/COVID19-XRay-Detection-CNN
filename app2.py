import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import keras
import sys
import os
import gdown
import os
import requests

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout



# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="COVID-19 X-Ray Detection",
    page_icon="🩺",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* Background */
.stApp{
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #1e3a8a 50%,
        #2563eb 100%
    );
}

/* Title */
.main-title{
    text-align:center;
    color:white;
    font-size:60px;
    font-weight:800;
    text-shadow:0px 3px 10px rgba(0,0,0,0.4);
}

/* Subtitle */
.sub-title{
    text-align:center;
    color:#dbeafe;
    font-size:22px;
    margin-bottom:20px;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background: linear-gradient(
        180deg,
        #0f172a,
        #1d4ed8,
        #3b82f6
    );
}

section[data-testid="stSidebar"] *{
    color:white !important;
}

/* Buttons */
.stButton > button{
    width:100%;
    border-radius:15px;
    height:55px;
    font-size:18px;
    font-weight:bold;
    border:none;
    color:white;
    background:linear-gradient(
        90deg,
        #06b6d4,
        #2563eb
    );
}

/* Metric cards */
[data-testid="metric-container"]{
    background:rgba(255,255,255,0.08);
    border-radius:15px;
    padding:10px;
    border:1px solid rgba(255,255,255,0.1);
}

/* File uploader */
[data-testid="stFileUploader"]{
    background:rgba(255,255,255,0.08);
    border-radius:15px;
    padding:10px;
}
/* Force text white */

h1,h2,h3,h4,h5,h6{
    color:white !important;
}



label{
    color:white !important;
}


</style>
""", unsafe_allow_html=True)

# =====================================================
# MODEL LOADING
# =====================================================

# =====================================================
# MODEL LOADING
# =====================================================

import os
import gdown

@st.cache_resource
def load_model():

    MODEL_PATH = "covid_weights.weights.h5"

    if not os.path.exists(MODEL_PATH):

        file_id = "18laJh_n-tjoDvp1v77TDjiBv_3-e1AaX"

        url = f"https://drive.google.com/uc?id={file_id}"

        gdown.download(url, MODEL_PATH, quiet=False)

    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224,224,3)
    )

    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.load_weights(MODEL_PATH)

    return model
model = load_model()

# =====================================================
# CLASS LABELS
# =====================================================

class_names = [
    "Covid",
    "Normal",
    "Viral Pneumonia"
]

# =====================================================
# IMAGE PREPROCESSING
# =====================================================

def preprocess_image(image):

    image = image.resize((224,224))

    img = np.array(image)

    if len(img.shape) == 2:
        img = np.stack([img]*3, axis=-1)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    return img

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class='main-title'>
🩺 COVID-19 X-Ray Detection System
</div>

<div class='sub-title'>
Deep Learning Based Disease Classification using VGG16 Transfer Learning
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🩺 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Prediction",
        "📊 Model Results"
    ]
)

# =====================================================
# PREDICTION PAGE
# =====================================================

if page == "🏠 Prediction":
    st.markdown("""
<div style="
background:rgba(255,255,255,0.15);
padding:20px;
border-radius:15px;
color:white;
font-size:18px;
margin-bottom:20px;
">

📤 Upload a chest X-Ray image and the AI model will classify it as
COVID-19, Normal, or Viral Pneumonia.

</div>
""", unsafe_allow_html=True)

    st.subheader("📤 Upload Chest X-Ray Image")

    uploaded_file = st.file_uploader(
        "Choose X-Ray Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Uploaded X-Ray",
                use_container_width=True
            )

        with col2:

            if st.button("🔍 Predict Disease"):

                processed_img = preprocess_image(image)

                prediction = model.predict(processed_img)

                pred_index = np.argmax(prediction)

                confidence = np.max(prediction) * 100

                predicted_class = class_names[pred_index]

                st.subheader("🎯 Prediction Result")

                if predicted_class == "Covid":

                    st.markdown(f"""
                                <div style="
                                background:linear-gradient(135deg,#dc2626,#b91c1c);
                                padding:25px;
                                border-radius:18px;
                                color:white;
                                box-shadow:0px 8px 25px rgba(220,38,38,0.4);
                                border:2px solid rgba(255,255,255,0.15);
                                ">

                                <h2 style="margin:0;">🚨 COVID DETECTED</h2>

                                <h3 style="margin-top:15px;">
                                Confidence: {confidence:.2f}%
                                </h3>

                                </div>
                                """, unsafe_allow_html=True)
                                

                                

                                            
 


                elif predicted_class == "Normal":

                    st.success(
                        f"✅ NORMAL CHEST X-RAY\n\nConfidence: {confidence:.2f}%"
                    )

                else:

                    st.warning(
                        f"⚠️ VIRAL PNEUMONIA DETECTED\n\nConfidence: {confidence:.2f}%"
                    )

                st.markdown("""
                <h1 style="
                color:white;
                margin-top:30px;
                ">
                Confidence Score
                </h1>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="
                background:linear-gradient(135deg,#06b6d4,#2563eb);
                padding:25px;
                border-radius:20px;
                box-shadow:0px 10px 25px rgba(37,99,235,0.4);
                border:2px solid rgba(255,255,255,0.15);
                text-align:center;
                margin-bottom:20px;
                ">

               <h3 style="color:white;margin:0;">
               Model Confidence
               </h3>

               <h1 style="
               color:white;
               font-size:55px;
               margin-top:10px;
               margin-bottom:10px;
               ">
               {confidence:.2f}%
               </h1>

               </div>
               """, unsafe_allow_html=True)

                st.subheader("📊 Prediction Probabilities")

                prob_df = pd.DataFrame({
                    "Class": class_names,
                    "Probability (%)": prediction[0] * 100
                })

                st.dataframe(
                    prob_df,
                    use_container_width=True
                )

                st.bar_chart(
                    prob_df.set_index("Class")
                )

# =====================================================
# RESULTS PAGE
# =====================================================

elif page == "📊 Model Results":

    st.subheader("📌 Project Summary")
    st.markdown("""
    <div style="
    background:rgba(255,255,255,0.15);
    padding:25px;
    border-radius:20px;
    color:white;
    font-size:20px;
    line-height:2;
    border:1px solid rgba(255,255,255,0.15);
    ">

   🎯 <b>Objective</b><br>
   Detect COVID-19, Normal and Viral Pneumonia from Chest X-Ray Images using Deep Learning.
   <br>

   🧠 <b>Model Used:</b> VGG16 Transfer Learning

   <br>

   📂 <b>Dataset:</b> COVID-19 Image Dataset

   <br>

   🖼 <b>Input Size:</b> 224 × 224 × 3

   <br>

   📊 <b>Test Samples:</b> 66 Images

   </div>
   """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("📈 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "98.48%")
    col2.metric("Loss", "0.0888")
    col3.metric("ROC-AUC", "0.9971")
    col4.metric("F1 Score", "0.98")

    st.markdown("---")

    st.subheader("📋 Classification Report")

    report_df = pd.DataFrame({
        "Class":[
            "Covid",
            "Normal",
            "Viral Pneumonia"
        ],
        "Precision":[1.00,1.00,0.95],
        "Recall":[1.00,0.95,1.00],
        "F1-Score":[1.00,0.97,0.98]
    })

    st.dataframe(
        report_df,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("🎯 Confusion Matrix")

    cm = np.array([
        [26,0,0],
        [0,19,1],
        [0,0,20]
    ])

    fig, ax = plt.subplots(figsize=(8,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        linewidths=2,
        linecolor='white',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    st.pyplot(fig)

    st.markdown("---")

    st.subheader("🧠 Why VGG16?")

    st.success("""
✔ Pre-trained on ImageNet

✔ Strong Feature Extraction

✔ Faster Convergence

✔ Excellent Performance on Medical Images

✔ Achieved 98.48% Test Accuracy

✔ Transfer Learning Reduced Training Time
""")

    st.markdown("---")

    st.subheader("📂 Dataset Information")

    st.info("""
Dataset Source:
https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset

Classes:
• Covid
• Normal
• Viral Pneumonia

Input Size:
• 224 × 224 × 3

Model:
• VGG16 Transfer Learning
""")

    st.markdown("---")

    st.warning("""
⚠ Disclaimer

This application is intended for educational and research purposes only.

It should not be used as a substitute for professional medical diagnosis, treatment, or clinical decision-making.
""")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown("""
<div style='text-align:center;color:white;'>

### 👨‍💻 Developed By Archit Tomar

AI • Machine Learning • Deep Learning • Computer Vision



</div>
""", unsafe_allow_html=True)

