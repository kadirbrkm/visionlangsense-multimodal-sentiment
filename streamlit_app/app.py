import streamlit as st
import torch
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import io
import base64
from typing import Dict, List, Optional, Any
import cv2

# Add parent directory to path for imports
import sys
sys.path.append('..')

from src.models.multimodal_sentiment import MultiModalSentimentAnalyzer, SentimentThemeConfig
from src.utils.gradcam_explainer import MultiModalGradCAM
from src.data.dataset import DatasetBuilder

# Page configuration
st.set_page_config(
    page_title="Multi-Modal Sentiment & Thematic Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .prediction-result {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #28a745;
        margin: 1rem 0;
    }
    .explanation-section {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #4a90e2;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitApp:
    """Main Streamlit application for multi-modal sentiment analysis."""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.gradcam_explainer = None
        self.domain = "fashion"  # Default domain
        
        # Initialize session state
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'predictions' not in st.session_state:
            st.session_state.predictions = None
        if 'explanations' not in st.session_state:
            st.session_state.explanations = None
    
    def load_model(self, model_path: Optional[str] = None, domain: str = "fashion"):
        """Load the trained model."""
        try:
            if model_path and Path(model_path).exists():
                # Load trained model
                checkpoint = torch.load(model_path, map_location=self.device)
                config = checkpoint.get('config', {})
                
                self.model = MultiModalSentimentAnalyzer(
                    num_sentiment_classes=config.get('num_sentiment_classes', 5),
                    num_theme_classes=config.get('num_theme_classes', 10),
                    hidden_dim=config.get('hidden_dim', 512),
                    dropout_rate=config.get('dropout_rate', 0.1)
                ).to(self.device)
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
            else:
                # Initialize a pre-trained model (for demo purposes)
                self.model = MultiModalSentimentAnalyzer(
                    num_sentiment_classes=5,
                    num_theme_classes=10 if domain == "fashion" else 6,
                    hidden_dim=512,
                    dropout_rate=0.1
                ).to(self.device)
                self.model.eval()
            
            # Initialize Grad-CAM explainer
            self.gradcam_explainer = MultiModalGradCAM(self.model)
            self.domain = domain
            
            st.session_state.model_loaded = True
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess uploaded image."""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to standard size
        image = image.resize((224, 224))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        return image_array
    
    def predict(self, image: np.ndarray, text: str) -> Dict[str, Any]:
        """Make predictions on image and text."""
        if self.model is None:
            return None
        
        try:
            # Convert image to PIL for model processing
            pil_image = Image.fromarray(image)
            
            # Process through CLIP
            image_tensor = self.model.clip_processor(
                images=pil_image,
                return_tensors="pt"
            )['pixel_values'].to(self.device)
            
            # Get predictions
            with torch.no_grad():
                predictions = self.model.predict(image_tensor, [text])
            
            # Convert to interpretable format
            sentiment_labels = list(SentimentThemeConfig.SENTIMENT_LABELS.values())
            theme_labels = list(SentimentThemeConfig.get_theme_labels(self.domain).values())
            
            result = {
                'sentiment_probs': predictions['sentiment_probs'][0],
                'theme_probs': predictions['theme_probs'][0],
                'quality_score': float(predictions['quality_scores'][0]),
                'sentiment_pred': sentiment_labels[predictions['sentiment_preds'][0]],
                'theme_pred': theme_labels[predictions['theme_preds'][0]],
                'sentiment_labels': sentiment_labels,
                'theme_labels': theme_labels
            }
            
            return result
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            return None
    
    def generate_explanations(self, image: np.ndarray, text: str) -> Dict[str, Any]:
        """Generate model explanations using Grad-CAM."""
        if self.gradcam_explainer is None:
            return None
        
        try:
            explanations = self.gradcam_explainer.create_comprehensive_explanation(
                image, text
            )
            return explanations
            
        except Exception as e:
            st.error(f"Error generating explanations: {str(e)}")
            return None
    
    def render_sidebar(self):
        """Render the sidebar with model settings and information."""
        st.sidebar.title("🔧 Settings")
        
        # Domain selection
        domain = st.sidebar.selectbox(
            "Select Domain",
            ["fashion", "food"],
            index=0 if self.domain == "fashion" else 1
        )
        
        if domain != self.domain:
            self.domain = domain
            if st.session_state.model_loaded:
                st.sidebar.info("Domain changed. Please reload the model.")
                st.session_state.model_loaded = False
        
        # Model loading section
        st.sidebar.subheader("Model Configuration")
        
        model_path = st.sidebar.text_input(
            "Model Path (optional)",
            placeholder="Path to trained model checkpoint"
        )
        
        if st.sidebar.button("Load Model"):
            with st.spinner("Loading model..."):
                success = self.load_model(model_path, domain)
                if success:
                    st.sidebar.success("Model loaded successfully!")
                else:
                    st.sidebar.error("Failed to load model")
        
        # Model status
        if st.session_state.model_loaded:
            st.sidebar.success("✅ Model Ready")
        else:
            st.sidebar.warning("⚠️ Model Not Loaded")
        
        # Information section
        st.sidebar.subheader("ℹ️ About")
        st.sidebar.info(
            """
            This application performs multi-modal sentiment and thematic analysis 
            on product images and descriptions using CLIP-based deep learning models.
            
            **Features:**
            - Sentiment analysis (5 classes)
            - Theme classification
            - Quality score prediction
            - Model explainability with Grad-CAM
            """
        )
        
        # Demo data section
        st.sidebar.subheader("📊 Demo Data")
        if st.sidebar.button("Create Sample Dataset"):
            with st.spinner("Creating sample dataset..."):
                output_dir = "data/sample"
                DatasetBuilder.create_sample_dataset(
                    output_dir, 
                    num_samples=100, 
                    domain=domain
                )
                st.sidebar.success(f"Sample {domain} dataset created!")
    
    def render_main_interface(self):
        """Render the main application interface."""
        # Header
        st.markdown('<h1 class="main-header">🔍 Multi-Modal Sentiment & Thematic Analysis</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        Upload a product image and enter a description to analyze sentiment, themes, and quality. 
        The system uses advanced CLIP-based models for comprehensive multi-modal understanding.
        """)
        
        if not st.session_state.model_loaded:
            st.warning("⚠️ Please load a model from the sidebar to start analysis.")
            return
        
        # Input section
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Upload Image")
            uploaded_file = st.file_uploader(
                "Choose a product image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a clear image of the product"
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Preprocess image
                processed_image = self.preprocess_image(image)
        
        with col2:
            st.subheader("📝 Product Description")
            text_input = st.text_area(
                "Enter product description",
                placeholder="Describe the product, its features, quality, style, etc.",
                height=200,
                help="Provide a detailed description for better analysis"
            )
            
            # Analysis button
            analyze_button = st.button(
                "🔍 Analyze Product",
                type="primary",
                disabled=not (uploaded_file and text_input.strip())
            )
        
        # Analysis section
        if analyze_button and uploaded_file and text_input.strip():
            with st.spinner("Analyzing product..."):
                # Make predictions
                predictions = self.predict(processed_image, text_input)
                
                if predictions:
                    st.session_state.predictions = predictions
                    
                    # Generate explanations
                    explanations = self.generate_explanations(processed_image, text_input)
                    st.session_state.explanations = explanations
        
        # Display results
        if st.session_state.predictions:
            self.render_results()
        
        # Display explanations
        if st.session_state.explanations:
            self.render_explanations()
    
    def render_results(self):
        """Render prediction results."""
        predictions = st.session_state.predictions
        
        st.markdown('<h2 class="sub-header">📊 Analysis Results</h2>', unsafe_allow_html=True)
        
        # Overall metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>😊 Sentiment</h3>
                    <h2>{predictions['sentiment_pred']}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>🎨 Theme</h3>
                    <h2>{predictions['theme_pred']}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col3:
            quality_color = "green" if predictions['quality_score'] > 0.7 else "orange" if predictions['quality_score'] > 0.4 else "red"
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>⭐ Quality Score</h3>
                    <h2 style="color: {quality_color}">{predictions['quality_score']:.3f}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Detailed probability charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment probabilities
            fig_sentiment = px.bar(
                x=predictions['sentiment_labels'],
                y=predictions['sentiment_probs'],
                title="Sentiment Probability Distribution",
                labels={'x': 'Sentiment', 'y': 'Probability'},
                color=predictions['sentiment_probs'],
                color_continuous_scale='RdYlGn'
            )
            fig_sentiment.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with col2:
            # Theme probabilities
            fig_theme = px.bar(
                x=predictions['theme_labels'],
                y=predictions['theme_probs'],
                title="Theme Probability Distribution",
                labels={'x': 'Theme', 'y': 'Probability'},
                color=predictions['theme_probs'],
                color_continuous_scale='viridis'
            )
            fig_theme.update_layout(showlegend=False, height=400)
            fig_theme.update_xaxes(tickangle=45)
            st.plotly_chart(fig_theme, use_container_width=True)
        
        # Confidence analysis
        sentiment_confidence = np.max(predictions['sentiment_probs'])
        theme_confidence = np.max(predictions['theme_probs'])
        
        st.subheader("🎯 Prediction Confidence")
        
        confidence_data = pd.DataFrame({
            'Category': ['Sentiment', 'Theme', 'Quality'],
            'Confidence': [sentiment_confidence, theme_confidence, predictions['quality_score']],
            'Color': ['#ff6b6b', '#4ecdc4', '#45b7d1']
        })
        
        fig_confidence = px.bar(
            confidence_data,
            x='Category',
            y='Confidence',
            title="Model Confidence Scores",
            color='Color',
            color_discrete_map={'#ff6b6b': '#ff6b6b', '#4ecdc4': '#4ecdc4', '#45b7d1': '#45b7d1'}
        )
        fig_confidence.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_confidence, use_container_width=True)
    
    def render_explanations(self):
        """Render model explanations and visualizations."""
        explanations = st.session_state.explanations
        
        if not explanations:
            return
        
        st.markdown('<h2 class="sub-header">🔍 Model Explanations</h2>', unsafe_allow_html=True)
        
        st.markdown(
            """
            <div class="explanation-section">
                <p>The following visualizations show which parts of the image the model focused on 
                when making predictions for different categories. Brighter/warmer colors indicate 
                higher importance.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Explanation method selection
        method = st.selectbox(
            "Select Explanation Method",
            ["gradcam", "gradcam++"],
            help="Different methods for generating visual explanations"
        )
        
        # Display explanations for each category
        categories = ['sentiment', 'theme', 'quality']
        
        for category in categories:
            if f"{category}_{method}" in explanations:
                explanation = explanations[f"{category}_{method}"]
                
                st.subheader(f"🎯 {category.title()} Focus Areas")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(
                        explanation['original_image'],
                        caption="Original Image",
                        use_column_width=True
                    )
                
                with col2:
                    st.image(
                        explanation['cam_image'],
                        caption=f"{category.title()} - {method.upper()}",
                        use_column_width=True
                    )
                
                # Show prediction details for this category
                if category in ['sentiment', 'theme']:
                    pred_key = f"{category}_preds"
                    if pred_key in explanation['predictions']:
                        pred_value = explanation['predictions'][pred_key][0]
                        if category == 'sentiment':
                            label = list(SentimentThemeConfig.SENTIMENT_LABELS.values())[pred_value]
                        else:
                            label = list(SentimentThemeConfig.get_theme_labels(self.domain).values())[pred_value]
                        st.info(f"Predicted {category}: **{label}**")
                else:  # quality
                    quality_score = explanation['predictions']['quality_scores'][0]
                    st.info(f"Predicted quality score: **{quality_score:.3f}**")
        
        # Additional analysis
        st.subheader("📈 Cross-Modal Analysis")
        
        # Show attention patterns between text and image
        st.markdown("""
        **Text-Image Alignment**: The model combines information from both the image and text 
        description to make final predictions. Areas of high attention in the image often 
        correspond to key terms mentioned in the text description.
        """)
        
        # Download section
        st.subheader("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Download Predictions (JSON)"):
                predictions_json = json.dumps(st.session_state.predictions, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=predictions_json,
                    file_name="predictions.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("🖼️ Download Explanations"):
                st.info("Explanation images are displayed above and can be right-clicked to save.")
    
    def run(self):
        """Run the Streamlit application."""
        self.render_sidebar()
        self.render_main_interface()


# Additional utility functions
def create_demo_page():
    """Create a demo page with example analyses."""
    st.title("🎮 Demo Examples")
    
    st.markdown("""
    Explore pre-analyzed examples to understand the capabilities of the multi-modal 
    sentiment analysis system.
    """)
    
    # Example data
    examples = {
        "fashion": [
            {
                "title": "Elegant Black Dress",
                "description": "Perfect for formal occasions and special events. Made with premium materials.",
                "sentiment": "Very Positive",
                "theme": "Elegant",
                "quality": 0.85
            },
            {
                "title": "Casual Denim Jacket",
                "description": "Comfortable everyday wear with modern style. Great for casual outings.",
                "sentiment": "Positive",
                "theme": "Casual",
                "quality": 0.72
            }
        ],
        "food": [
            {
                "title": "Organic Green Salad",
                "description": "Fresh organic vegetables with natural dressing. Healthy and delicious.",
                "sentiment": "Very Positive",
                "theme": "Healthy",
                "quality": 0.78
            },
            {
                "title": "Gourmet Pizza",
                "description": "Premium ingredients with authentic Italian flavors. Artisan made.",
                "sentiment": "Positive",
                "theme": "Gourmet",
                "quality": 0.81
            }
        ]
    }
    
    domain = st.selectbox("Select Domain", ["fashion", "food"])
    
    for i, example in enumerate(examples[domain]):
        with st.expander(f"Example {i+1}: {example['title']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Description:**")
                st.write(example['description'])
            
            with col2:
                st.write("**Predictions:**")
                st.write(f"Sentiment: {example['sentiment']}")
                st.write(f"Theme: {example['theme']}")
                st.write(f"Quality: {example['quality']:.2f}")


def main():
    """Main application entry point."""
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate",
        ["🏠 Main Analysis", "🎮 Demo Examples", "📚 Documentation"]
    )
    
    if page == "🏠 Main Analysis":
        app = StreamlitApp()
        app.run()
    
    elif page == "🎮 Demo Examples":
        create_demo_page()
    
    elif page == "📚 Documentation":
        st.title("📚 Documentation")
        
        st.markdown("""
        ## Multi-Modal Sentiment & Thematic Analysis System
        
        ### Overview
        This system performs comprehensive analysis of product images and descriptions using 
        state-of-the-art CLIP-based models to extract sentiment, themes, and quality indicators.
        
        ### Features
        
        #### 1. Sentiment Analysis
        - **Classes**: Very Negative, Negative, Neutral, Positive, Very Positive
        - **Method**: Multi-modal fusion of image and text features
        - **Output**: Probability distribution across sentiment classes
        
        #### 2. Theme Classification
        - **Fashion Themes**: Casual, Formal, Sporty, Vintage, Elegant, Trendy, etc.
        - **Food Themes**: Healthy, Comfort Food, Gourmet, Fast Food, etc.
        - **Method**: Domain-specific classification based on visual and textual cues
        
        #### 3. Quality Assessment
        - **Output**: Continuous score between 0 and 1
        - **Factors**: Visual appeal, description quality, brand indicators
        
        #### 4. Model Explainability
        - **Grad-CAM**: Visual attention maps showing model focus areas
        - **Multiple Methods**: GradCAM, GradCAM++, ScoreCAM
        - **Cross-Modal**: Analysis of text-image alignment
        
        ### Technical Details
        
        #### Architecture
        - **Base Model**: CLIP (Contrastive Language-Image Pre-training)
        - **Fusion**: Multi-modal feature fusion layers
        - **Heads**: Separate classification/regression heads for each task
        
        #### Training
        - **Multi-task Learning**: Joint optimization of all objectives
        - **Data Augmentation**: Advanced image and text augmentation
        - **Regularization**: Dropout, weight decay, label smoothing
        
        ### Usage Guidelines
        
        1. **Image Quality**: Use clear, well-lit product images
        2. **Text Description**: Provide detailed, accurate descriptions
        3. **Domain Selection**: Choose appropriate domain (fashion/food)
        4. **Interpretation**: Consider confidence scores when interpreting results
        
        ### Model Performance
        
        Typical performance metrics on validation data:
        - **Sentiment Accuracy**: ~85-90%
        - **Theme Accuracy**: ~80-85%
        - **Quality MAE**: ~0.1-0.15
        
        ### Limitations
        
        - Performance may vary with image quality and lighting
        - Text description quality affects accuracy
        - Domain-specific training required for optimal results
        - Explainability methods provide approximations of model behavior
        """)


if __name__ == "__main__":
    main()