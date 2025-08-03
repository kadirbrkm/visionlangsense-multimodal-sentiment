import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from typing import List, Optional, Tuple, Dict, Any
import matplotlib.patches as patches


class MultiModalGradCAM:
    """
    Enhanced Grad-CAM implementation for multi-modal sentiment analysis model.
    Provides explainability for both image and text modalities.
    """
    
    def __init__(self, model, target_layers: Optional[List] = None):
        """
        Initialize Grad-CAM explainer.
        
        Args:
            model: The multi-modal sentiment analyzer model
            target_layers: List of target layers for CAM generation
        """
        self.model = model
        self.device = next(model.parameters()).device
        
        # If no target layers specified, use the last convolutional layer of CLIP vision model
        if target_layers is None:
            # For CLIP ViT, we use the last transformer block
            target_layers = [model.clip_model.vision_model.encoder.layers[-1].layer_norm1]
        
        self.target_layers = target_layers
        
        # Initialize different CAM methods
        self.cam_methods = {
            'gradcam': GradCAM(model=model, target_layers=target_layers),
            'gradcam++': GradCAMPlusPlus(model=model, target_layers=target_layers),
            'scorecam': ScoreCAM(model=model, target_layers=target_layers),
            'xgradcam': XGradCAM(model=model, target_layers=target_layers),
            'ablationcam': AblationCAM(model=model, target_layers=target_layers),
            'eigencam': EigenCAM(model=model, target_layers=target_layers)
        }
    
    def generate_cam(self, 
                     image: np.ndarray,
                     text: str,
                     method: str = 'gradcam',
                     target_category: Optional[str] = None,
                     target_class: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate CAM heatmap for a given image-text pair.
        
        Args:
            image: Input image as numpy array (RGB)
            text: Input text description
            method: CAM method to use ('gradcam', 'gradcam++', etc.)
            target_category: Category to explain ('sentiment', 'theme', 'quality')
            target_class: Specific class index to target
            
        Returns:
            Dictionary containing CAM visualization and related information
        """
        self.model.eval()
        
        # Preprocess image for the model
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
            
        # Convert to tensor and add batch dimension
        image_tensor = self.model.clip_processor(
            images=pil_image, 
            return_tensors="pt"
        )['pixel_values'].to(self.device)
        
        # Create a custom target for multi-modal model
        class MultiModalTarget:
            def __init__(self, category, class_idx=None):
                self.category = category
                self.class_idx = class_idx
                
            def __call__(self, model_output):
                if self.category == 'sentiment':
                    logits = model_output['sentiment_logits']
                elif self.category == 'theme':
                    logits = model_output['theme_logits']
                elif self.category == 'quality':
                    return model_output['quality_scores']
                else:
                    logits = model_output['sentiment_logits']  # Default
                
                if self.class_idx is not None:
                    return logits[:, self.class_idx]
                else:
                    return logits.max(1)[0]
        
        # Create wrapper model for GradCAM
        class ModelWrapper(torch.nn.Module):
            def __init__(self, original_model, text_input):
                super().__init__()
                self.model = original_model
                self.text = text_input
                
            def forward(self, x):
                return self.model(x, [self.text])
        
        wrapped_model = ModelWrapper(self.model, text)
        
        # Get CAM method
        cam = self.cam_methods[method]
        cam.model = wrapped_model
        
        # Define target
        if target_category and target_class is not None:
            targets = [MultiModalTarget(target_category, target_class)]
        else:
            targets = None
            
        # Generate CAM
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
        
        # Convert image to RGB numpy array for visualization
        rgb_image = np.array(pil_image) / 255.0
        
        # Generate visualization
        cam_image = show_cam_on_image(rgb_image, grayscale_cam[0], use_rgb=True)
        
        # Get model predictions for context
        with torch.no_grad():
            predictions = self.model.predict(image_tensor, [text])
        
        return {
            'cam_image': cam_image,
            'grayscale_cam': grayscale_cam[0],
            'original_image': rgb_image,
            'predictions': predictions,
            'method': method,
            'target_category': target_category,
            'target_class': target_class
        }
    
    def create_comprehensive_explanation(self, 
                                       image: np.ndarray,
                                       text: str,
                                       save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a comprehensive explanation with multiple CAM methods and targets.
        
        Args:
            image: Input image
            text: Input text description
            save_path: Optional path to save the visualization
            
        Returns:
            Dictionary containing all explanations and visualizations
        """
        explanations = {}
        
        # Get model predictions first
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
            
        image_tensor = self.model.clip_processor(
            images=pil_image, 
            return_tensors="pt"
        )['pixel_values'].to(self.device)
        
        with torch.no_grad():
            predictions = self.model.predict(image_tensor, [text])
        
        explanations['predictions'] = predictions
        
        # Generate CAMs for different categories and top predictions
        categories = ['sentiment', 'theme', 'quality']
        methods = ['gradcam', 'gradcam++']
        
        for category in categories:
            if category == 'quality':
                # For quality, we don't need class-specific CAM
                for method in methods:
                    key = f"{category}_{method}"
                    explanations[key] = self.generate_cam(
                        image, text, method=method, target_category=category
                    )
            else:
                # Get top predicted class for this category
                if category == 'sentiment':
                    top_class = predictions['sentiment_preds'][0]
                else:  # theme
                    top_class = predictions['theme_preds'][0]
                
                for method in methods:
                    key = f"{category}_{method}"
                    explanations[key] = self.generate_cam(
                        image, text, method=method, 
                        target_category=category, target_class=int(top_class)
                    )
        
        # Create comprehensive visualization
        if save_path:
            self._create_comprehensive_plot(explanations, save_path)
        
        return explanations
    
    def _create_comprehensive_plot(self, explanations: Dict[str, Any], save_path: str):
        """Create a comprehensive plot with all explanations."""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Multi-Modal Sentiment Analysis - Model Explanations', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(explanations['sentiment_gradcam']['original_image'])
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Sentiment explanations
        axes[0, 1].imshow(explanations['sentiment_gradcam']['cam_image'])
        axes[0, 1].set_title('Sentiment - GradCAM')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(explanations['sentiment_gradcam++']['cam_image'])
        axes[0, 2].set_title('Sentiment - GradCAM++')
        axes[0, 2].axis('off')
        
        # Theme explanations
        axes[1, 0].imshow(explanations['theme_gradcam']['cam_image'])
        axes[1, 0].set_title('Theme - GradCAM')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(explanations['theme_gradcam++']['cam_image'])
        axes[1, 1].set_title('Theme - GradCAM++')
        axes[1, 1].axis('off')
        
        # Quality explanations
        axes[1, 2].imshow(explanations['quality_gradcam']['cam_image'])
        axes[1, 2].set_title('Quality - GradCAM')
        axes[1, 2].axis('off')
        
        # Prediction summaries
        predictions = explanations['predictions']
        
        # Sentiment prediction chart
        from src.models.multimodal_sentiment import SentimentThemeConfig
        sentiment_labels = list(SentimentThemeConfig.SENTIMENT_LABELS.values())
        sentiment_probs = predictions['sentiment_probs'][0]
        
        axes[2, 0].bar(range(len(sentiment_labels)), sentiment_probs)
        axes[2, 0].set_xticks(range(len(sentiment_labels)))
        axes[2, 0].set_xticklabels(sentiment_labels, rotation=45, ha='right')
        axes[2, 0].set_title('Sentiment Predictions')
        axes[2, 0].set_ylabel('Probability')
        
        # Theme prediction chart
        theme_labels = list(SentimentThemeConfig.FASHION_THEMES.values())
        theme_probs = predictions['theme_probs'][0]
        
        axes[2, 1].bar(range(len(theme_labels)), theme_probs)
        axes[2, 1].set_xticks(range(len(theme_labels)))
        axes[2, 1].set_xticklabels(theme_labels, rotation=45, ha='right')
        axes[2, 1].set_title('Theme Predictions')
        axes[2, 1].set_ylabel('Probability')
        
        # Quality score
        quality_score = predictions['quality_scores'][0]
        axes[2, 2].bar(['Quality Score'], [quality_score])
        axes[2, 2].set_ylim(0, 1)
        axes[2, 2].set_title(f'Quality Score: {quality_score:.3f}')
        axes[2, 2].set_ylabel('Score')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


class TextAttentionVisualizer:
    """
    Visualize attention patterns in the text modality.
    """
    
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
    
    def visualize_text_attention(self, 
                                text: str,
                                image: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Visualize attention patterns in text processing.
        
        Args:
            text: Input text to analyze
            image: Optional image for multi-modal context
            
        Returns:
            Dictionary containing attention visualizations
        """
        self.model.eval()
        
        # Tokenize text
        inputs = self.model.clip_processor(
            text=[text], 
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Get tokens
        tokens = self.model.clip_processor.tokenizer.convert_ids_to_tokens(
            inputs['input_ids'][0]
        )
        
        # If image provided, get multimodal embeddings
        if image is not None:
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
                
            inputs = self.model.clip_processor(
                text=[text],
                images=pil_image,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get attention weights (requires model modification to return attention)
        # This is a simplified version - in practice, you'd need to modify CLIP
        # to return attention weights
        
        with torch.no_grad():
            if image is not None:
                outputs = self.model.forward(
                    pil_image if isinstance(pil_image, torch.Tensor) else [pil_image], 
                    [text]
                )
            else:
                # Text-only analysis
                clip_outputs = self.model.clip_model.get_text_features(**inputs)
        
        return {
            'tokens': tokens,
            'text': text,
            'attention_weights': None,  # Would need model modification
            'embeddings': clip_outputs if 'clip_outputs' in locals() else None
        }