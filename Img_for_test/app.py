import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import plotly.graph_objects as go

# Define categories
CATEGORIES = ['Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy']

# Disease information
DISEASE_INFO = {
    'Potato_Early_blight': {
        'description': 'Early blight is caused by the fungus Alternaria solani.',
        'symptoms': ['Dark brown spots with rings', 'Yellow tissue around spots', 'Early leaf drop'],
        'treatment': ['Remove infected plants', 'Apply fungicides', 'Ensure good air circulation'],
        'severity': 'Moderate',
        'color': 'orange'
    },
    'Potato_Late_blight': {
        'description': 'Late blight is caused by Phytophthora infestans.',
        'symptoms': ['Water-soaked spots', 'White fungal growth', 'Rapid spread'],
        'treatment': ['Use resistant varieties', 'Apply fungicides', 'Destroy infected plants'],
        'severity': 'High',
        'color': 'red'
    },
    'Potato_healthy': {
        'description': 'The plant is healthy with no signs of disease.',
        'symptoms': ['No symptoms'],
        'treatment': ['Continue regular monitoring'],
        'severity': 'None',
        'color': 'green'
    }
}

# Page config
st.set_page_config(
    page_title="Potato Disease Detection",
    page_icon="🥔",
    layout="wide"
)

# Title
st.title("🥔 Potato Disease Detection System")
st.markdown("---")

@st.cache_resource
def load_models():
    """Load the models"""
    inception_model = None
    cnn_model = None
    
    try:
        # Load Inception model
        if os.path.exists('Models/inception_savedmodel'):
            try:
                inception_model = tf.keras.models.load_model('Models/inception_savedmodel')
            except:
                inception_model = tf.saved_model.load('Models/inception_savedmodel')
        
        # Load CNN model
        if os.path.exists('Models/cnn_savedmodel'):
            try:
                cnn_model = tf.keras.models.load_model('Models/cnn_savedmodel')
            except:
                cnn_model = tf.saved_model.load('Models/cnn_savedmodel')
                
    except Exception as e:
        st.error(f"Error loading models: {e}")
    
    return inception_model, cnn_model

def preprocess_image(image):
    """Preprocess image for model"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

def make_prediction(model, processed_image, model_name):
    """Make prediction"""
    try:
        if hasattr(model, 'predict'):
            predictions = model.predict(processed_image, verbose=0)
        else:
            infer = model.signatures['serving_default']
            input_tensor = tf.constant(processed_image)
            predictions_dict = infer(input_tensor)
            predictions = list(predictions_dict.values())[0].numpy()
        
        predicted_idx = np.argmax(predictions[0])
        predicted_class = CATEGORIES[predicted_idx]
        confidence = float(predictions[0][predicted_idx] * 100)
        
        all_probs = [float(predictions[0][i] * 100) for i in range(3)]
        
        return predicted_class, confidence, all_probs
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None

# Load models
inception_model, cnn_model = load_models()

if not inception_model and not cnn_model:
    st.error("❌ No models found!")
    st.stop()

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔍 Detection", "📊 Model Performance", "📚 Disease Info"])

# Tab 1: Detection
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader("Choose a potato leaf image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Model selection
            models = []
            if inception_model:
                models.append("InceptionV3")
            if cnn_model:
                models.append("CNN")
            if len(models) > 1:
                models.append("Both")
            
            selected = st.radio("Select Model:", models)
            
            if st.button("🔍 Analyze", type="primary", use_container_width=True):
                processed = preprocess_image(image)
                st.session_state.predictions = []
                
                if selected in ["InceptionV3", "Both"] and inception_model:
                    pred_class, conf, probs = make_prediction(inception_model, processed, "InceptionV3")
                    if pred_class:
                        st.session_state.predictions.append({
                            'model': 'InceptionV3',
                            'class': pred_class,
                            'confidence': conf,
                            'probs': probs
                        })
                
                if selected in ["CNN", "Both"] and cnn_model:
                    pred_class, conf, probs = make_prediction(cnn_model, processed, "CNN")
                    if pred_class:
                        st.session_state.predictions.append({
                            'model': 'CNN',
                            'class': pred_class,
                            'confidence': conf,
                            'probs': probs
                        })
    
    with col2:
        st.header("📊 Results")
        
        if 'predictions' in st.session_state and st.session_state.predictions:
            for pred in st.session_state.predictions:
                st.subheader(f"**{pred['model']} Model**")
                
                # Display result
                disease = pred['class'].replace('_', ' ')
                color = DISEASE_INFO[pred['class']]['color']
                
                if 'healthy' in pred['class'].lower():
                    st.success(f"✅ {disease}")
                elif 'Late' in disease:
                    st.error(f"🦠 {disease}")
                else:
                    st.warning(f"⚠️ {disease}")
                
                st.write(f"**Confidence:** {pred['confidence']:.2f}%")
                
                # Simple bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Early Blight', 'Late Blight', 'Healthy'],
                        y=pred['probs'],
                        marker_color=['orange', 'red', 'green'],
                        text=[f'{p:.1f}%' for p in pred['probs']],
                        textposition='auto'
                    )
                ])
                fig.update_layout(
                    title=f"{pred['model']} Predictions",
                    yaxis_title="Probability (%)",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show disease info
                if pred['class'] in DISEASE_INFO:
                    info = DISEASE_INFO[pred['class']]
                    with st.expander("View Details"):
                        st.write(f"**Description:** {info['description']}")
                        st.write(f"**Severity:** {info['severity']}")
                        st.write("**Treatment:**")
                        for t in info['treatment']:
                            st.write(f"- {t}")
                
                st.markdown("---")
        else:
            st.info("👈 Upload an image and click Analyze")

# Tab 2: Model Performance
with tab2:
    st.header("📊 Confusion Matrices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("InceptionV3 Model")
        if os.path.exists('Data/confusion_Inception.png'):
            img = Image.open('Data/confusion_Inception.png')
            st.image(img, use_container_width=True)
        else:
            st.warning("Confusion matrix not found at Data/confusion_Inception.png")
    
    with col2:
        st.subheader("CNN Model")
        if os.path.exists('Data/confusion_CNN.png'):
            img = Image.open('Data/confusion_CNN.png')
            st.image(img, use_container_width=True)
        else:
            st.warning("Confusion matrix not found at Data/confusion_CNN.png")

# Tab 3: Disease Information
with tab3:
    st.header("📚 Disease Information")
    
    for disease, info in DISEASE_INFO.items():
        name = disease.replace('_', ' ')
        
        with st.expander(f"🥔 {name}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Description:** {info['description']}")
                st.write(f"**Severity:** {info['severity']}")
                st.write("**Symptoms:**")
                for s in info['symptoms']:
                    st.write(f"• {s}")
            
            with col2:
                st.write("**Treatment:**")
                for t in info['treatment']:
                    st.write(f"• {t}")
                
                if info['severity'] == 'High':
                    st.error("⚠️ Immediate action required!")
                elif info['severity'] == 'Moderate':
                    st.warning("⚠️ Monitor closely")
                else:
                    st.success("✅ Healthy")