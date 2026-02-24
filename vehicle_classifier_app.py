import streamlit as st
from PIL import Image
import numpy as np
import random
#import tensorflow as tf

MODEL_PATH = 'vehicle_model.h5'

# Use st.cache_resource to load the model once
'''@st.cache_resource
def get_model():
   
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Load the model
model = get_model()'''

# Page configuration
st.set_page_config(
    page_title="Vehicle Classifier",
    page_icon="🚗",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to bottom right, #0f172a, #1e293b, #0f172a);
    }
    .stButton>button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
    .result-card {
        background-color: #334155;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #475569;
        margin: 1rem 0;
    }
    .probability-bar {
        height: 12px;
        border-radius: 9999px;
        margin: 0.25rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

# Vehicle classes and their colors
VEHICLE_CLASSES = ['Bus', 'Car', 'Motorcycle', 'Light Truck', 'Heavy Truck']
CLASS_COLORS = {
    'Bus': '#3b82f6',
    'Car': '#22c55e',
    'Motorcycle': '#a855f7',
    'Light Truck': '#f97316',
    'Heavy Truck': '#ef4444'
}

def simple_image_classifier(image):
    """
    Simple classifier based on image characteristics
    This is a demonstration - in production you'd use a real ML model
    """
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Get image dimensions
    height, width = img_array.shape[:2]
    aspect_ratio = width / height if height > 0 else 1
    
    # Calculate average color values
    avg_color = img_array.mean(axis=(0, 1))
    
    # Simple heuristic-based classification
    # In a real app, this would be replaced with a trained ML model
    
    # Initialize probabilities
    probabilities = {class_name: 5 for class_name in VEHICLE_CLASSES}
    
    # Use image characteristics to adjust probabilities
    # These are simplified heuristics for demonstration
    
    # Aspect ratio heuristics
    if aspect_ratio > 1.8:  # Wide images might be buses or trucks
        probabilities['Bus'] += 20
        probabilities['Heavy Truck'] += 15
    elif aspect_ratio < 1.2:  # Taller images might be motorcycles
        probabilities['Motorcycle'] += 25
    else:  # Medium aspect ratio
        probabilities['Car'] += 20
        probabilities['Light Truck'] += 15
    
    # Brightness heuristics (simplified)
    brightness = avg_color.mean()
    if brightness > 150:  # Bright images
        probabilities['Car'] += 10
    
    # Add some randomness for variety
    for class_name in VEHICLE_CLASSES:
        probabilities[class_name] += random.randint(-5, 15)
    
    # Ensure all probabilities are non-negative
    for class_name in VEHICLE_CLASSES:
        probabilities[class_name] = max(5, probabilities[class_name])
    
    # Normalize to sum to 100
    total = sum(probabilities.values())
    probabilities = {k: round((v / total) * 100) for k, v in probabilities.items()}
    
    # Adjust to ensure sum is exactly 100
    diff = 100 - sum(probabilities.values())
    if diff != 0:
        max_class = max(probabilities.items(), key=lambda x: x[1])[0]
        probabilities[max_class] += diff
    
    return probabilities

def get_top_class(results):
    """Get the class with highest probability"""
    return max(results.items(), key=lambda x: x[1])[0]

def display_probability_bar(class_name, probability):
    """Display a custom probability bar"""
    color = CLASS_COLORS.get(class_name, '#6b7280')
    st.markdown(f"""
        <div style="margin: 0.5rem 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                <span style="color: #cbd5e1; font-size: 0.875rem; font-weight: 500;">{class_name}</span>
                <span style="color: #e2e8f0; font-size: 0.875rem; font-weight: 600;">{probability}%</span>
            </div>
            <div style="width: 100%; background-color: #334155; border-radius: 9999px; height: 12px; overflow: hidden;">
                <div style="width: {probability}%; background-color: {color}; height: 100%; border-radius: 9999px; transition: width 0.5s;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Header
st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: black; font-size: 2.5rem; margin-bottom: 0.5rem;">🚗 Vehicle Classifier</h1>
        <p style="color: black;">Upload an image to classify the vehicle type</p>
    </div>
""", unsafe_allow_html=True)

# Info box
st.info("ℹ️ This is a vehicle classification app. Upload an image of a vehicle, and the app will analyze it to determine the type of vehicle along with confidence levels. ")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=['png', 'jpg', 'jpeg', 'gif'],
    help="Upload a vehicle image (PNG, JPG, GIF up to 10MB)"
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Store image in session state
    st.session_state.uploaded_image = uploaded_file
    
    # Classify button
    if st.button("🔍 Classify Vehicle"):
        with st.spinner("Analyzing vehicle..."):
            # Classify the image
            results = simple_image_classifier(image)
            
            if results:
                st.session_state.results = results
    
    # Display results
    if st.session_state.results:
        results = st.session_state.results
        top_class = get_top_class(results)
        
        # Top classification result
        st.markdown(f"""
            <div class="result-card">
                <h2 style="color: white; font-size: 1.5rem; margin-bottom: 0.5rem;">
                    Classification: <span style="color: #60a5fa;">{top_class}</span>
                </h2>
                <p style="color: #cbd5e1; font-size: 0.875rem;">
                    Confidence: {results[top_class]}%
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # All probabilities
        st.markdown("""
            <h3 style="color: white; font-size: 1.125rem; font-weight: 600; margin-top: 1.5rem; margin-bottom: 1rem;">
                All Probabilities:
            </h3>
        """, unsafe_allow_html=True)
        
        for class_name in VEHICLE_CLASSES:
            display_probability_bar(class_name, results[class_name])
        
        # Clear button
        if st.button("🔄 Classify Another Image"):
            st.session_state.results = None
            st.session_state.uploaded_image = None
            st.rerun()
else:
    st.info("👆 Upload a vehicle image to get started!")
    
    # Reset results when no file is uploaded
    if st.session_state.results:
        st.session_state.results = None