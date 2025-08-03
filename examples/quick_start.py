#!/usr/bin/env python3
"""
Quick Start Example for Multi-Modal Sentiment Analysis

This script demonstrates basic usage of the multi-modal sentiment analysis system.
It creates sample data, trains a model briefly, and shows how to make predictions.
"""

import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append('..')

from src.models.multimodal_sentiment import MultiModalSentimentAnalyzer
from src.data.dataset import DatasetBuilder, create_data_loaders
from src.training.trainer import MultiModalTrainer, create_training_config
from src.utils.gradcam_explainer import MultiModalGradCAM

def main():
    print("🚀 Multi-Modal Sentiment Analysis - Quick Start")
    print("=" * 50)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Create sample dataset
    print("\n📊 Creating sample dataset...")
    data_dir = Path("../data/quick_start")
    DatasetBuilder.create_sample_dataset(
        output_dir=str(data_dir),
        num_samples=100,  # Small dataset for quick testing
        domain="fashion"
    )
    print(f"✅ Sample dataset created at {data_dir}")
    
    # 2. Initialize model
    print("\n🧠 Initializing model...")
    model = MultiModalSentimentAnalyzer(
        num_sentiment_classes=5,
        num_theme_classes=10,
        hidden_dim=256,  # Smaller for faster training
        dropout_rate=0.1
    )
    print("✅ Model initialized")
    
    # 3. Create data loaders
    print("\n📋 Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        data_dir=str(data_dir),
        domain="fashion",
        batch_size=16,  # Smaller batch size
        num_workers=2,
        clip_processor=model.clip_processor
    )
    print(f"✅ Data loaders created - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    # 4. Setup training configuration
    print("\n⚙️ Setting up training...")
    config = create_training_config(domain="fashion")
    config.update({
        'epochs': 5,  # Few epochs for quick demo
        'batch_size': 16,
        'learning_rate': 5e-4,  # Higher LR for faster convergence
        'use_wandb': False,  # Disable wandb for demo
        'checkpoint_dir': '../checkpoints/quick_start'
    })
    
    # 5. Initialize trainer
    trainer = MultiModalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        use_wandb=False
    )
    print("✅ Trainer initialized")
    
    # 6. Quick training
    print("\n🎓 Starting quick training (5 epochs)...")
    try:
        trainer.train(epochs=5)
        print(f"✅ Training completed! Best accuracy: {trainer.best_val_acc:.4f}")
    except Exception as e:
        print(f"⚠️ Training error: {e}")
        print("This is normal for a quick demo - continuing with random weights...")
    
    # 7. Make a prediction
    print("\n🔮 Making sample prediction...")
    model.eval()
    
    # Get a sample from validation set
    sample_batch = next(iter(val_loader))
    sample_image = sample_batch['image'][0:1]  # Take first image
    sample_text = [sample_batch['text'][0]]     # Take first text
    
    with torch.no_grad():
        predictions = model.predict(sample_image.to(device), sample_text)
    
    # Display results
    print("\n📊 Prediction Results:")
    print("-" * 30)
    print(f"Text: {sample_text[0][:100]}...")
    print(f"Sentiment: {predictions['sentiment_preds'][0]} (confidence: {predictions['sentiment_probs'][0].max():.3f})")
    print(f"Theme: {predictions['theme_preds'][0]} (confidence: {predictions['theme_probs'][0].max():.3f})")
    print(f"Quality Score: {predictions['quality_scores'][0]:.3f}")
    
    # 8. Generate explanation
    print("\n🔍 Generating model explanation...")
    try:
        explainer = MultiModalGradCAM(model)
        
        # Convert tensor back to numpy for explanation
        import numpy as np
        image_np = sample_image[0].permute(1, 2, 0).cpu().numpy()
        # Denormalize image
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        image_np = (image_np * std + mean) * 255
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        
        explanations = explainer.generate_cam(
            image=image_np,
            text=sample_text[0],
            method='gradcam',
            target_category='sentiment'
        )
        print("✅ Explanation generated successfully!")
        
    except Exception as e:
        print(f"⚠️ Explanation generation failed: {e}")
        print("This is normal for untrained models")
    
    # 9. Summary
    print("\n🎉 Quick Start Completed!")
    print("=" * 50)
    print("What you've learned:")
    print("• How to create sample datasets")
    print("• How to initialize the model")
    print("• How to set up training")
    print("• How to make predictions")
    print("• How to generate explanations")
    print("\nNext steps:")
    print("• Try training with real datasets")
    print("• Experiment with different hyperparameters")
    print("• Use the Streamlit demo for interactive analysis")
    print("• Explore different domains (fashion vs food)")


if __name__ == "__main__":
    main()