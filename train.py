#!/usr/bin/env python3
"""
Main training script for Multi-Modal Sentiment Analysis project.

This script sets up the complete training pipeline including:
- Data loading and preprocessing
- Model initialization 
- Training with comprehensive logging
- Model evaluation and saving

Usage:
    python train.py --domain fashion --epochs 100 --batch_size 32
    python train.py --config configs/fashion_config.json
"""

import argparse
import json
import torch
import wandb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.models.multimodal_sentiment import MultiModalSentimentAnalyzer
from src.data.dataset import create_data_loaders, DatasetBuilder
from src.training.trainer import MultiModalTrainer, create_training_config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Multi-Modal Sentiment Analysis Model")
    
    # Data arguments
    parser.add_argument('--domain', type=str, default='fashion', 
                       choices=['fashion', 'food'],
                       help='Product domain to train on')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing the dataset')
    parser.add_argument('--create_sample_data', action='store_true',
                       help='Create sample dataset for testing')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples in generated dataset')
    
    # Model arguments
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-base-patch32',
                       help='CLIP model to use as base')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension for fusion layers')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--clip_lr', type=float, default=1e-5,
                       help='Learning rate for CLIP parameters')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for regularization')
    
    # Loss weights
    parser.add_argument('--sentiment_weight', type=float, default=1.0,
                       help='Weight for sentiment loss')
    parser.add_argument('--theme_weight', type=float, default=1.0,
                       help='Weight for theme loss')
    parser.add_argument('--quality_weight', type=float, default=0.5,
                       help='Weight for quality loss')
    
    # Technical arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use automatic mixed precision')
    
    # Experiment tracking
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for the experiment')
    parser.add_argument('--use_wandb', action='store_true', default=True,
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='multimodal-sentiment-analysis',
                       help='Weights & Biases project name')
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Directory to save checkpoints')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Configuration file
    parser.add_argument('--config', type=str, default=None,
                       help='Path to JSON configuration file')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_config_from_args(args) -> dict:
    """Create configuration dictionary from command line arguments."""
    config = {
        # Model parameters
        'clip_model_name': args.clip_model,
        'hidden_dim': args.hidden_dim,
        'dropout_rate': args.dropout_rate,
        'num_sentiment_classes': 5,
        'num_theme_classes': 10 if args.domain == 'fashion' else 6,
        
        # Training parameters
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'clip_learning_rate': args.clip_lr,
        'head_learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        
        # Loss weights
        'sentiment_weight': args.sentiment_weight,
        'theme_weight': args.theme_weight,
        'quality_weight': args.quality_weight,
        
        # Technical parameters
        'use_amp': args.use_amp,
        'num_workers': args.num_workers,
        'pin_memory': True,
        
        # Domain and data
        'domain': args.domain,
        'data_dir': args.data_dir,
        
        # Experiment tracking
        'experiment_name': args.experiment_name,
        'use_wandb': args.use_wandb,
        'wandb_project': args.wandb_project,
        
        # Checkpointing
        'checkpoint_dir': args.checkpoint_dir or f'checkpoints/{args.domain}_sentiment_analysis',
        'save_every': 10,
    }
    
    return config


def setup_device(device_arg: str) -> str:
    """Setup and return the device to use."""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Load or create configuration
    if args.config:
        print(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        # Override with command line arguments where provided
        for key, value in vars(args).items():
            if value is not None and key != 'config':
                config[key] = value
    else:
        config = create_config_from_args(args)
    
    # Setup device
    device = setup_device(args.device)
    
    # Create sample dataset if requested
    if args.create_sample_data:
        print(f"Creating sample {args.domain} dataset...")
        DatasetBuilder.create_sample_dataset(
            output_dir=Path(args.data_dir) / "sample",
            num_samples=args.num_samples,
            domain=args.domain
        )
        config['data_dir'] = str(Path(args.data_dir) / "sample")
    
    # Initialize model
    print("Initializing model...")
    model = MultiModalSentimentAnalyzer(
        clip_model_name=config['clip_model_name'],
        num_sentiment_classes=config['num_sentiment_classes'],
        num_theme_classes=config['num_theme_classes'],
        hidden_dim=config['hidden_dim'],
        dropout_rate=config['dropout_rate']
    )
    
    # Create data loaders
    print("Creating data loaders...")
    try:
        train_loader, val_loader = create_data_loaders(
            data_dir=config['data_dir'],
            domain=config['domain'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            clip_processor=model.clip_processor
        )
        
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        print("Make sure your dataset is properly formatted or use --create_sample_data")
        return
    
    # Initialize trainer
    print("Setting up trainer...")
    trainer = MultiModalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        use_wandb=config['use_wandb'],
        experiment_name=config['experiment_name']
    )
    
    # Print training summary
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION SUMMARY")
    print("="*50)
    print(f"Domain: {config['domain']}")
    print(f"Model: {config['clip_model_name']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Device: {device}")
    print(f"Mixed Precision: {config['use_amp']}")
    print(f"Checkpoint Dir: {config['checkpoint_dir']}")
    if config['use_wandb']:
        print(f"W&B Project: {config['wandb_project']}")
    print("="*50)
    
    # Start training
    try:
        trainer.train(
            epochs=config['epochs'],
            resume_from=args.resume_from
        )
        
        print("\n🎉 Training completed successfully!")
        print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
        print(f"Checkpoints saved to: {config['checkpoint_dir']}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        print("Saving current checkpoint...")
        trainer.save_checkpoint(trainer.val_metrics[-1] if trainer.val_metrics else {})
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise
    
    finally:
        if config['use_wandb']:
            wandb.finish()


if __name__ == "__main__":
    main()