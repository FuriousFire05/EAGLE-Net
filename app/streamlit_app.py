"""
Streamlit demo app for EAGLE-Net inference.
Run: streamlit run app/streamlit_app.py
"""

import streamlit as st
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.models.architectures import create_model
from src.inference.predictor import Predictor
from src.data.dataset import EUROSAT_CLASSES


# Page config
st.set_page_config(
    page_title='EAGLE-Net Demo',
    page_icon='🦅',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.1rem;
        padding: 0.5rem 1rem;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
    }
    h2 {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_predictor(model_name, checkpoint_path, device):
    """Load model and create predictor (cached)."""
    class_names = list(EUROSAT_CLASSES.values())
    
    model = create_model(model_name, num_classes=len(class_names))
    model.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    predictor = Predictor(model, class_names, device=device)
    
    return predictor, class_names


def format_probability(prob):
    """Format probability as percentage."""
    return f"{prob * 100:.1f}%"


def main():
    """Main app function."""
    
    # Header
    st.markdown("# 🦅 EAGLE-Net: EuroSAT Land Classification")
    st.markdown(
        "**Efficient Attention for Geo-spatial Land Estimation Network**  \n"
        "Multi-class satellite imagery classification with PyTorch + Streamlit"
    )
    
    # Sidebar settings
    st.sidebar.markdown("## ⚙️ Settings")
    
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.sidebar.info(f"💻 Device: **{device}**")
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "📊 Select Model",
        options=['eager_net', 'lightweight_cnn', 'baseline_cnn'],
        help="Different architectures with varying complexity"
    )
    
    # Checkpoint path
    checkpoint_dir = Path('artifacts')
    checkpoint_path = checkpoint_dir / 'best_model.pt'
    
    if not checkpoint_path.exists():
        st.error(f"❌ Model checkpoint not found at `{checkpoint_path}`")
        st.info("Please train a model first using: `python train.py`")
        return
    
    # Load model
    try:
        predictor, class_names = load_model_and_predictor(
            model_name, str(checkpoint_path), device
        )
        st.sidebar.success("✓ Model loaded successfully")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🖼️ Upload Image", "📊 Dataset Info", "ℹ️ About"])
    
    # Tab 1: Upload and Predict
    with tab1:
        st.markdown("### Upload EuroSAT Image")
        
        # Upload widget
        uploaded_file = st.file_uploader(
            "Choose an image (JPG, PNG)",
            type=['jpg', 'jpeg', 'png'],
            help="64×64 RGB satellite image"
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Input Image")
                st.image(image, use_column_width=True, caption="Uploaded Image")
            
            # Predict
            with st.spinner("🔮 Making prediction..."):
                try:
                    result = predictor.predict_from_array(image)
                    
                    # Get top-3
                    probs = result['probabilities']
                    top_indices = np.argsort(probs)[-3:][::-1]
                    top_classes = [class_names[i] for i in top_indices]
                    top_probs = probs[top_indices]
                    
                    with col2:
                        st.markdown("##### 🎯 Top-3 Predictions")
                        
                        # Medal emojis
                        medals = ['🥇', '🥈', '🥉']
                        
                        for medal, cls_name, prob in zip(medals, top_classes, top_probs):
                            st.markdown(
                                f"<div class='prediction-box'>"
                                f"{medal} <b>{cls_name}</b><br>"
                                f"Confidence: <b>{format_probability(prob)}</b>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                    
                    # Detailed predictions
                    st.markdown("---")
                    st.markdown("### 📈 All Class Predictions")
                    
                    # Sorted by probability
                    sorted_indices = np.argsort(probs)[::-1]
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    colors = ['#1f77b4' if i == sorted_indices[0] else '#d3d3d3' for i in sorted_indices]
                    bars = ax.barh(
                        [class_names[i] for i in sorted_indices],
                        probs[sorted_indices],
                        color=colors,
                        edgecolor='navy',
                        alpha=0.8
                    )
                    
                    # Add value labels
                    for i, (idx, bar) in enumerate(zip(sorted_indices, bars)):
                        prob = probs[idx]
                        ax.text(prob + 0.01, i, f'{prob:.1%}', va='center', fontweight='bold')
                    
                    ax.set_xlabel('Probability', fontsize=11, fontweight='bold')
                    ax.set_title('Classification Probabilities', fontsize=12, fontweight='bold')
                    ax.set_xlim([0, 1])
                    ax.grid(True, alpha=0.3, axis='x')
                    
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    
                    # Metrics
                    st.markdown("---")
                    st.markdown("### 📊 Prediction Metrics")
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric(
                            "Primary Prediction",
                            top_classes[0],
                            f"{format_probability(top_probs[0])} confidence"
                        )
                    
                    with col_b:
                        st.metric(
                            "Entropy",
                            f"{-np.sum(probs * np.log(probs + 1e-10)):.3f}",
                            "Measure of uncertainty"
                        )
                    
                    with col_c:
                        st.metric(
                            "Top-1 vs Top-2 Gap",
                            f"{format_probability(top_probs[0] - top_probs[1])}",
                            "Confidence margin"
                        )
                
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")
        
        else:
            st.info("👆 Upload an image to get started")
    
    # Tab 2: Dataset Info
    with tab2:
        st.markdown("### 🌍 EuroSAT Dataset Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Dataset Overview**")
            st.write(
                """
                - **Name:** EuroSAT
                - **Classes:** 10 land-use types
                - **Total Images:** ~27,000
                - **Image Size:** 64×64 pixels
                - **Bands:** RGB (3 channels)
                - **Source:** Sentinel-2 satellite data
                """
            )
        
        with col2:
            st.markdown("**Class Distribution**")
            st.write(
                """
                1. Annual Crop
                2. Forest
                3. Herbaceous Vegetation
                4. Highway
                5. Industrial
                6. Pasture
                7. Permanent Crop
                8. Residential
                9. River
                10. Sea Lake
                """
            )
        
        st.markdown("---")
        
        # Class info
        st.markdown("### 📋 Class Details")
        
        class_info = {
            'Annual Crop': 'Farmland with seasonal crops',
            'Forest': 'Densely vegetated forest areas',
            'Herbaceous Vegetation': 'Non-tree vegetation',
            'Highway': 'Major roads and expressways',
            'Industrial': 'Industrial and commercial zones',
            'Pasture': 'Grassland for livestock',
            'Permanent Crop': 'Vineyards, orchards',
            'Residential': 'Urban residential areas',
            'River': 'Inland water bodies',
            'Sea Lake': 'Large water bodies',
        }
        
        for class_name, description in class_info.items():
            with st.expander(f"📍 {class_name}"):
                st.write(description)
    
    # Tab 3: About
    with tab3:
        st.markdown("### 🦅 About EAGLE-Net")
        
        st.markdown(
            """
            **EAGLE-Net: Efficient Attention for Geo-spatial Land Estimation Network**
            
            A lightweight deep learning model for satellite image classification, optimized for
            EuroSAT land-use classification task.
            
            #### 🏗️ Architecture Features
            
            - **Backbone:** Depthwise separable convolutions (parameter-efficient)
            - **Attention:** Channel + Spatial attention modules (CBAM-style)
            - **Parameters:** ~100K (vs ~1M for standard CNN)
            - **Inference:** Real-time on CPU
            
            #### 🎯 Model Variants
            
            1. **BaselineCNN:** Standard CNN baseline (~500K params)
            2. **LightweightCNN:** Efficient depthwise separable (~100K params)
            3. **EAGLENet:** Lightweight + Channel/Spatial Attention (~150K params)
            
            #### 📊 Performance
            
            - **Best Val Accuracy:** ~95% on EuroSAT
            - **Inference Speed:** <100ms per image (CPU)
            - **Model Size:** ~1-2 MB
            
            #### 🛠️ Technology Stack
            
            - **Framework:** PyTorch
            - **Data:** torchvision transforms
            - **UI:** Streamlit
            - **Metrics:** scikit-learn
            - **Visualization:** Matplotlib, Seaborn
            
            #### 📚 Project Structure
            
            ```
            EAGLE-Net/
            ├── src/
            │   ├── data/        (dataset loading)
            │   ├── models/      (architectures)
            │   ├── training/    (trainer, loops)
            │   ├── inference/   (predictor)
            │   └── utils/       (metrics, viz)
            ├── app/             (Streamlit demo)
            ├── artifacts/       (saved models)
            └── train.py         (training entry point)
            ```
            
            #### 🚀 Quick Start
            
            ```bash
            # 1. Install dependencies
            pip install -r requirements.txt
            
            # 2. Download EuroSAT dataset
            
            # 3. Train model
            python train.py --model eager_net --epochs 100
            
            # 4. Run Streamlit demo
            streamlit run app/streamlit_app.py
            ```
            
            #### 📖 For More Info
            
            See `README.md` in project root for detailed documentation.
            """
        )
        
        st.markdown("---")
        st.markdown("**Built for deep learning assignments. Educational project.**")


if __name__ == '__main__':
    main()
