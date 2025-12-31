import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import matplotlib.pyplot as plt
import io

# Page config
st.set_page_config(
    page_title="Leukemia Detection AI",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f1f5f9;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
    .warning-box {
        background-color: #fef3c7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model (cached)
@st.cache_resource
def load_model():
    """Load trained ResNet50 model"""
    try:
        # Update this path to your model location
        model = keras.models.load_model('resnet50_best.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please upload your resnet50_best.h5 model file to the app directory")
        return None

# Grad-CAM function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name='conv5_block3_3_conv'):
    """Generate Grad-CAM heatmap"""
    base_model = model.layers[0]
    
    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    last_conv_layer_model = Model(base_model.input, last_conv_layer.output)
    
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    x = model.layers[1](x)
    x = model.layers[2](x)
    x = model.layers[3](x)
    x = model.layers[4](x)
    classifier_model = Model(classifier_input, x)
    
    with tf.GradientTape() as tape:
        conv_outputs = last_conv_layer_model(img_array)
        tape.watch(conv_outputs)
        predictions = classifier_model(conv_outputs)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def create_gradcam_overlay(img, heatmap, alpha=0.4):
    """Overlay heatmap on original image"""
    heatmap = np.uint8(255 * heatmap)
    
    jet = plt.colormaps.get_cmap('jet')
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img

def preprocess_image(image):
    """Preprocess uploaded image for model"""
    img = image.resize((224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Main app
def main():
    # Header
    st.markdown('<p class="main-header">🔬 Leukemia Detection AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep Learning-Based Blood Cell Classification with Explainable AI</p>', unsafe_allow_html=True)
    
    # Warning disclaimer
    st.markdown("""
    <div class="warning-box">
        ⚠️ <strong>Medical Disclaimer:</strong> This tool is for research and educational purposes only. 
        It should NOT be used as a substitute for professional medical diagnosis. 
        Always consult qualified healthcare professionals for medical decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar - Model Info
    with st.sidebar:
        st.header("📊 Model Information")
        st.markdown("""
        **Architecture:** ResNet50 Transfer Learning
        
        **Training Dataset:** C-NMC Leukemia (10,661 images)
        
        **Performance:**
        - Accuracy: 90.6%
        - Recall: 95.0%
        - Precision: 89.4%
        
        **Classes:**
        - Normal cells (hem)
        - Leukemia cells (all)
        """)
        
        st.markdown("---")
        st.header("📖 How to Use")
        st.markdown("""
        1. Upload blood cell image (BMP/PNG/JPG)
        2. Click "Analyze Image"
        3. View prediction + confidence
        4. Examine Grad-CAM heatmap
        """)
        
        st.markdown("---")
        st.header("🎨 Heatmap Guide")
        st.markdown("""
        - 🔴 **Red/Yellow**: High attention
        - 🟢 **Cyan/Green**: Medium attention
        - 🔵 **Blue/Purple**: Low attention
        
        Model focuses on nucleus regions for leukemia detection.
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a blood cell image...",
            type=['bmp', 'png', 'jpg', 'jpeg'],
            help="Upload a microscopy image of a blood cell"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Analyze button
            if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Preprocess
                    img_array = preprocess_image(image)
                    
                    # Predict
                    pred = model.predict(img_array, verbose=0)
                    pred_prob = float(pred[0][0])
                    pred_class = int(pred_prob > 0.5)
                    confidence = pred_prob if pred_class == 1 else 1 - pred_prob
                    
                    # Generate Grad-CAM
                    heatmap = make_gradcam_heatmap(img_array, model)
                    
                    # Store in session state
                    st.session_state['prediction'] = {
                        'class': pred_class,
                        'confidence': confidence,
                        'prob': pred_prob,
                        'heatmap': heatmap,
                        'image': image
                    }
    
    with col2:
        st.header("📊 Results")
        
        if 'prediction' in st.session_state:
            result = st.session_state['prediction']
            
            # Prediction result
            pred_class = result['class']
            confidence = result['confidence']
            
            if pred_class == 1:
                st.error(f"🔴 **LEUKEMIA DETECTED**")
                st.metric("Confidence", f"{confidence*100:.1f}%")
            else:
                st.success(f"🟢 **NORMAL CELL**")
                st.metric("Confidence", f"{confidence*100:.1f}%")
            
            # Confidence bar
            st.progress(confidence)
            
            # Interpretation
            st.markdown("---")
            st.subheader("🧠 Model Interpretation (Grad-CAM)")
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["Heatmap", "Overlay", "Side-by-Side"])
            
            with tab1:
                # Heatmap only
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(result['heatmap'], cmap='jet')
                ax.axis('off')
                ax.set_title('Attention Heatmap', fontsize=14, fontweight='bold')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig)
                plt.close()
            
            with tab2:
                # Overlay
                img_array = keras.preprocessing.image.img_to_array(result['image'])
                overlay = create_gradcam_overlay(img_array, result['heatmap'], alpha=0.4)
                
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(overlay)
                ax.axis('off')
                ax.set_title('Grad-CAM Overlay', fontsize=14, fontweight='bold')
                st.pyplot(fig)
                plt.close()
            
            with tab3:
                # Side by side
                img_array = keras.preprocessing.image.img_to_array(result['image'])
                overlay = create_gradcam_overlay(img_array, result['heatmap'], alpha=0.4)
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                axes[0].imshow(result['image'])
                axes[0].set_title('Original', fontsize=12, fontweight='bold')
                axes[0].axis('off')
                
                axes[1].imshow(overlay)
                axes[1].set_title('Grad-CAM', fontsize=12, fontweight='bold')
                axes[1].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Explanation
            st.markdown("---")
            st.subheader("💡 What does this mean?")
            
            if pred_class == 1:
                st.markdown("""
                The model detected **abnormal characteristics** typical of leukemia:
                - Red/yellow regions show where model focused
                - Usually highlights irregular or enlarged nucleus
                - High attention on chromatin patterns
                
                ⚠️ **Action Required:** Consult hematologist for confirmation
                """)
            else:
                st.markdown("""
                The model identified **normal cell characteristics**:
                - Uniform cell structure
                - Regular nucleus shape
                - Normal nucleus-to-cytoplasm ratio
                
                ✅ **Note:** This indicates healthy cell morphology
                """)
        
        else:
            st.info("👆 Upload an image and click 'Analyze Image' to see results")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 2rem 0;'>
        <p><strong>Leukemia Detection AI</strong> | Built with ResNet50 Transfer Learning</p>
        <p>Dataset: C-NMC Leukemia (Gupta & Gupta, 2019) | Framework: TensorFlow/Keras</p>
        <p>For research and educational purposes only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
