import os
import numpy as np
import streamlit as st
from PIL import Image
import cv2

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Breast Cancer Image Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling for styling metrics, headers, and UI alerts
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
        text-align: left;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #6B7280;
        text-transform: uppercase;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #111827;
    }
    .result-box {
        border-radius: 12px;
        padding: 2rem;
        color: white;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .result-benign {
        background: linear-gradient(135deg, #10B981, #059669);
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.2);
    }
    .result-malignant {
        background: linear-gradient(135deg, #EF4444, #DC2626);
        box-shadow: 0 4px 6px -1px rgba(239, 68, 68, 0.2);
    }
    .technical-note {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<div class="main-title">🔬 Breast Cancer Histopathology Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deploying Model Breast Cancer Augmented for Clinical Image Inference</div>', unsafe_allow_html=True)

# ----------------- SIDEBAR CONFIGURATION -----------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model File Path configuration
    model_filename = "model_breast_cancer_augmented.h5"
    st.info(f"Using model: `{model_filename}`")
    
    # Adjustable Classification Threshold
    st.subheader("Threshold Fine-Tuning")
    st.write("Tune the sensitivity of the binary sigmoid output:")
    decision_threshold = st.slider(
        "Malignant Probability Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Adjust the classification boundary. Increasing this threshold reduces false positives but may increase false negatives."
    )
    
    st.markdown("---")
    st.markdown("### 🧑‍💻 Technical Summary")
    st.markdown("""
    **Model Architecture:**
    - Input Shape: `[250, 250, 3]`
    - Feature Extractor: Custom 3-layer CNN
    - Fully Connected: 512 -> 256 -> 128
    - Output Node: `Dense(1, activation='sigmoid')`
    - Normalization: `[0.0, 1.0]` (divided by 255)
    - Color Channel Order: **BGR** (OpenCV format)
    """)

# ----------------- MODEL LOADING FUNCTION -----------------
@st.cache_resource
def load_augmented_model(filepath: str):
    """
    Safely loads and caches the TensorFlow/Keras h5 model.
    """
    # Lazy import to avoid loading tensorflow if not required, speeding up initial boot
    import tensorflow as tf
    try:
        loaded_model = tf.keras.models.load_model(filepath)
        return loaded_model
    except Exception as error_msg:
        st.error(f"❌ Failed to load the model file from `{filepath}`. Details: {error_msg}")
        return None

# Attempt to load the model
model_path = os.path.join(os.getcwd(), model_filename)

if not os.path.exists(model_path):
    st.error(f"⚠️ Model file `{model_filename}` not found in the current directory.")
    st.stop()

# Show status loading model
with st.spinner("🔄 Initializing TensorFlow & Loading Classification Model..."):
    classification_model = load_augmented_model(model_path)

if classification_model is None:
    st.stop()

# ----------------- UI CONTROLS & FILE UPLOADER -----------------
col_upload, col_prediction = st.columns([1, 1])

with col_upload:
    st.subheader("📤 Upload Histopathology Image")
    uploaded_file = st.file_uploader(
        "Choose a histopathology image (400X zoom recommended)...",
        type=["png", "jpg", "jpeg"]
    )
    
    if uploaded_file is not None:
        # Load image via PIL
        pil_image = Image.open(uploaded_file)
        
        # Display the uploaded image
        # Using custom width='stretch' as per global rules
        st.image(
            pil_image, 
            caption="Uploaded Histopathology Tissue Image", 
            width="stretch"
        )
        
        st.success("✅ Image uploaded successfully.")
        
    else:
        st.info("ℹ️ Please upload a tissue patch image to run classification.")

# ----------------- PREPROCESSING & INFERENCE -----------------
with col_prediction:
    st.subheader("📊 Model Diagnostics & Inference")
    
    if uploaded_file is not None:
        with st.spinner("🤖 Running Preprocessing & Deep Learning Inference..."):
            
            # Step 1: Convert PIL RGB image to NumPy RGB
            rgb_array = np.array(pil_image)
            
            # Step 2: Ensure we have exactly 3 channels (RGB)
            if len(rgb_array.shape) == 2:
                # Convert Grayscale to RGB
                rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_GRAY2RGB)
            elif rgb_array.shape[2] == 4:
                # Remove Alpha Channel
                rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2RGB)
            
            # Step 3: Resize image to model expected shape (250, 250)
            resized_rgb = cv2.resize(rgb_array, (250, 250))
            
            # Step 4: Correct the channel order from RGB to BGR (essential as training was in BGR!)
            bgr_array = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
            
            # Step 5: Normalize image to [0.0, 1.0] (exactly as in training pipeline)
            normalized_image = bgr_array.astype("float32") / 255.0
            
            # Step 6: Expand dims to create batch dimension (1, 250, 250, 3)
            inference_batch = np.expand_dims(normalized_image, axis=0)
            
            # Step 7: Run Model Prediction
            raw_prediction = classification_model.predict(inference_batch)
            
            # Extract raw probability
            malignant_probability = float(raw_prediction[0][0])
            benign_probability = 1.0 - malignant_probability
            
            # Apply decision threshold to classify
            is_malignant = malignant_probability >= decision_threshold
            predicted_class_name = "Malignant" if is_malignant else "Benign"
            confidence_score = malignant_probability if is_malignant else benign_probability
            
            # Display Preprocessing details for auditability
            with st.expander("🛠️ Preprocessing Pipeline Inspection"):
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    # Width='stretch' strictly as per user global rules
                    st.image(resized_rgb, caption="1. Resized RGB Image (250x250)", width="stretch")
                with col_c2:
                    # To render BGR properly in Streamlit's RGB viewer, we convert it back or display differences
                    st.image(bgr_array, caption="2. BGR Color Matrix Swap", width="stretch")
                st.write(f"Inference Array Shape: `{inference_batch.shape}`")
                st.write(f"Pixel value bounds: `[{np.min(inference_batch):.4f}, {np.max(inference_batch):.4f}]`")
            
            # Display Results Cards
            if is_malignant:
                result_css = "result-box result-malignant"
                st.markdown(f"""
                <div class="{result_css}">
                    <h2>⚠️ Predicted Diagnosis: MALIGNANT</h2>
                    <p style="font-size: 1.2rem;">Model detected structural abnormalities consistent with malignant breast tissue.</p>
                    <p style="font-size: 1rem; opacity: 0.9;">Inference Probability: <strong>{malignant_probability*100:.2f}%</strong> (Threshold: {decision_threshold:.2f})</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                result_css = "result-box result-benign"
                st.markdown(f"""
                <div class="{result_css}">
                    <h2>✅ Predicted Diagnosis: BENIGN</h2>
                    <p style="font-size: 1.2rem;">Model suggests regular, non-malignant tissue characteristics.</p>
                    <p style="font-size: 1rem; opacity: 0.9;">Inference Probability: <strong>{benign_probability*100:.2f}%</strong> (Threshold: {decision_threshold:.2f})</p>
                </div>
                """, unsafe_allow_html=True)
                
            # Confidence Breakdown Columns
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid #10B981;">
                    <div class="metric-title">Benign Confidence</div>
                    <div class="metric-value">{benign_probability*100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_b2:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid #EF4444;">
                    <div class="metric-title">Malignant Confidence</div>
                    <div class="metric-value">{malignant_probability*100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            # Visualizing the decision boundary with a progress bar
            st.write("### 🧭 Decision Boundary Position")
            st.write("Visualizes the raw output probability relative to your threshold setting:")
            
            # Render a custom multi-colored slider-style progress bar using HTML
            bar_color = "#EF4444" if is_malignant else "#10B981"
            st.markdown(f"""
            <div style="background-color: #E5E7EB; border-radius: 6px; height: 16px; width: 100%; position: relative; margin-top: 10px; margin-bottom: 25px;">
                <!-- Threshold Marker -->
                <div style="position: absolute; left: {decision_threshold * 100}%; top: -6px; width: 4px; height: 28px; background-color: #1F2937; z-index: 10;" title="Threshold: {decision_threshold}">
                    <span style="position: absolute; top: -18px; left: -25px; font-size: 0.75rem; font-weight: bold; color: #1F2937;">Cutoff ({decision_threshold:.2f})</span>
                </div>
                <!-- Probability Fill -->
                <div style="background-color: {bar_color}; width: {malignant_probability * 100}%; height: 100%; border-radius: 6px; transition: width 0.5s ease-in-out;"></div>
                <!-- Underlay Text -->
                <div style="position: absolute; left: 2%; top: 18px; font-size: 0.75rem; color: #6B7280;">Benign Area (0.0)</div>
                <div style="position: absolute; right: 2%; top: 18px; font-size: 0.75rem; color: #6B7280;">Malignant Area (1.0)</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Clinical/Medical Disclaimer
            st.warning("""
            ⚠️ **Medical Disclaimer:** This tool is provided solely for educational, research, and developer demonstration purposes. 
            It is not intended to serve as a diagnostic tool, nor should it substitute for professional medical advice, evaluation, or consultation. 
            Always confirm diagnostic outputs with clinical pathologists and professional imaging criteria.
            """)
            
    else:
        st.write("📊 Upload an image from the left panel to display classifications and model diagnostics.")
        
        # Display instructions for testing
        st.markdown("""
        ### How to run inference:
        1. Locate histopathology tissue images (such as breast biopsy tissue slides, e.g., from the BreaKHis dataset).
        2. Drag and drop the `.png`, `.jpg`, or `.jpeg` file into the uploader on the left side of the panel.
        3. Adjust the threshold slider in the sidebar to see how the model's confidence maps to classifications.
        4. Review the **Preprocessing Pipeline Inspection** expansion pane to verify pixel values, resolution, and color swaps.
        """)
