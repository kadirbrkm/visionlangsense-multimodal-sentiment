import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from pathlib import Path
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from typing import Dict, List, Tuple, Optional, Any
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from ..models.multimodal_sentiment import MultiModalSentimentAnalyzer, SentimentThemeConfig
from ..data.dataset import create_data_loaders
from ..utils.gradcam_explainer import MultiModalGradCAM


class MultiModalTrainer:
    """
    Comprehensive trainer for multi-modal sentiment analysis model.
    Supports multiple loss functions, metrics, and advanced training techniques.
    """
    
    def __init__(self,
                 model: MultiModalSentimentAnalyzer,
                 train_loader,
                 val_loader,
                 config: Dict[str, Any],
                 device: str = 'cuda',
                 use_wandb: bool = True,
                 experiment_name: str = None):
        """
        Initialize the trainer.
        
        Args:
            model: Multi-modal sentiment analyzer model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dictionary
            device: Training device ('cuda' or 'cpu')
            use_wandb: Whether to use Weights & Biases for logging
            experiment_name: Name for the experiment
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        
        # Initialize training components
        self._setup_optimizers()
        self._setup_loss_functions()
        self._setup_schedulers()
        
        # Mixed precision training
        self.scaler = GradScaler()
        self.use_amp = config.get('use_amp', True)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        # Setup experiment tracking
        if self.use_wandb:
            self._setup_wandb(experiment_name)
        
        # Setup checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup explainability
        self.gradcam_explainer = MultiModalGradCAM(model)
    
    def _setup_wandb(self, experiment_name: str):
        """Setup Weights & Biases logging."""
        if experiment_name is None:
            experiment_name = f"multimodal_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            project="multimodal-sentiment-analysis",
            name=experiment_name,
            config=self.config,
            reinit=True
        )
        wandb.watch(self.model)
    
    def _setup_optimizers(self):
        """Setup optimizers for different model components."""
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        # Different learning rates for CLIP and classification heads
        clip_lr = self.config.get('clip_learning_rate', lr * 0.1)
        head_lr = self.config.get('head_learning_rate', lr)
        
        # Group parameters
        clip_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'clip_model' in name:
                clip_params.append(param)
            else:
                head_params.append(param)
        
        param_groups = [
            {'params': clip_params, 'lr': clip_lr, 'weight_decay': weight_decay},
            {'params': head_params, 'lr': head_lr, 'weight_decay': weight_decay}
        ]
        
        optimizer_name = self.config.get('optimizer', 'adamw')
        if optimizer_name.lower() == 'adamw':
            self.optimizer = optim.AdamW(param_groups)
        elif optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(param_groups)
        else:
            self.optimizer = optim.SGD(param_groups, momentum=0.9)
    
    def _setup_loss_functions(self):
        """Setup loss functions for multi-task learning."""
        # Get class weights if available
        try:
            class_weights = self.train_loader.dataset.get_class_weights()
            sentiment_weights = class_weights['sentiment'].to(self.device)
            theme_weights = class_weights['theme'].to(self.device)
        except:
            sentiment_weights = None
            theme_weights = None
        
        # Loss functions
        self.sentiment_criterion = nn.CrossEntropyLoss(weight=sentiment_weights)
        self.theme_criterion = nn.CrossEntropyLoss(weight=theme_weights)
        self.quality_criterion = nn.MSELoss()
        
        # Loss weights for multi-task learning
        self.loss_weights = {
            'sentiment': self.config.get('sentiment_weight', 1.0),
            'theme': self.config.get('theme_weight', 1.0),
            'quality': self.config.get('quality_weight', 1.0)
        }
    
    def _setup_schedulers(self):
        """Setup learning rate schedulers."""
        scheduler_type = self.config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.get('epochs', 100)
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            self.scheduler = None
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss."""
        losses = {}
        
        # Sentiment loss
        sentiment_loss = self.sentiment_criterion(
            outputs['sentiment_logits'], 
            batch['sentiment_label']
        )
        losses['sentiment'] = sentiment_loss
        
        # Theme loss
        theme_loss = self.theme_criterion(
            outputs['theme_logits'], 
            batch['theme_label']
        )
        losses['theme'] = theme_loss
        
        # Quality loss
        quality_loss = self.quality_criterion(
            outputs['quality_scores'], 
            batch['quality_score']
        )
        losses['quality'] = quality_loss
        
        # Total weighted loss
        total_loss = (
            self.loss_weights['sentiment'] * sentiment_loss +
            self.loss_weights['theme'] * theme_loss +
            self.loss_weights['quality'] * quality_loss
        )
        losses['total'] = total_loss
        
        return losses
    
    def compute_metrics(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute evaluation metrics."""
        metrics = {}
        
        # Convert to numpy for sklearn metrics
        sentiment_preds = torch.argmax(outputs['sentiment_logits'], dim=1).cpu().numpy()
        sentiment_true = batch['sentiment_label'].cpu().numpy()
        
        theme_preds = torch.argmax(outputs['theme_logits'], dim=1).cpu().numpy()
        theme_true = batch['theme_label'].cpu().numpy()
        
        quality_preds = outputs['quality_scores'].cpu().numpy()
        quality_true = batch['quality_score'].cpu().numpy()
        
        # Accuracy metrics
        metrics['sentiment_acc'] = accuracy_score(sentiment_true, sentiment_preds)
        metrics['theme_acc'] = accuracy_score(theme_true, theme_preds)
        
        # F1 scores
        metrics['sentiment_f1'] = f1_score(sentiment_true, sentiment_preds, average='weighted')
        metrics['theme_f1'] = f1_score(theme_true, theme_preds, average='weighted')
        
        # Quality metrics (MAE and RMSE)
        quality_mae = np.mean(np.abs(quality_preds - quality_true))
        quality_rmse = np.sqrt(np.mean((quality_preds - quality_true) ** 2))
        metrics['quality_mae'] = quality_mae
        metrics['quality_rmse'] = quality_rmse
        
        # Overall accuracy (average of sentiment and theme)
        metrics['overall_acc'] = (metrics['sentiment_acc'] + metrics['theme_acc']) / 2
        
        return metrics
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_losses = {'total': 0, 'sentiment': 0, 'theme': 0, 'quality': 0}
        total_metrics = {'sentiment_acc': 0, 'theme_acc': 0, 'sentiment_f1': 0, 'theme_f1': 0, 
                        'quality_mae': 0, 'quality_rmse': 0, 'overall_acc': 0}
        
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass with mixed precision
                with autocast(enabled=self.use_amp):
                    outputs = self.model(batch['image'], batch['text'])
                    losses = self.compute_loss(outputs, batch)
                
                # Backward pass
                if self.use_amp:
                    self.scaler.scale(losses['total']).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    losses['total'].backward()
                    self.optimizer.step()
                
                # Compute metrics
                with torch.no_grad():
                    metrics = self.compute_metrics(outputs, batch)
                
                # Accumulate losses and metrics
                for key in total_losses:
                    total_losses[key] += losses[key].item()
                
                for key in total_metrics:
                    total_metrics[key] += metrics[key]
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{losses['total'].item():.4f}",
                    'sent_acc': f"{metrics['sentiment_acc']:.3f}",
                    'theme_acc': f"{metrics['theme_acc']:.3f}"
                })
                
                # Log to wandb
                if self.use_wandb and batch_idx % 100 == 0:
                    wandb.log({
                        'train/step_loss': losses['total'].item(),
                        'train/step_sentiment_acc': metrics['sentiment_acc'],
                        'train/step_theme_acc': metrics['theme_acc'],
                        'train/step': self.current_epoch * num_batches + batch_idx
                    })
        
        # Average losses and metrics
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        return {**avg_losses, **avg_metrics}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_losses = {'total': 0, 'sentiment': 0, 'theme': 0, 'quality': 0}
        total_metrics = {'sentiment_acc': 0, 'theme_acc': 0, 'sentiment_f1': 0, 'theme_f1': 0, 
                        'quality_mae': 0, 'quality_rmse': 0, 'overall_acc': 0}
        
        num_batches = len(self.val_loader)
        all_sentiment_preds, all_sentiment_true = [], []
        all_theme_preds, all_theme_true = [], []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass
                outputs = self.model(batch['image'], batch['text'])
                losses = self.compute_loss(outputs, batch)
                metrics = self.compute_metrics(outputs, batch)
                
                # Accumulate losses and metrics
                for key in total_losses:
                    total_losses[key] += losses[key].item()
                
                for key in total_metrics:
                    total_metrics[key] += metrics[key]
                
                # Store predictions for detailed analysis
                sentiment_preds = torch.argmax(outputs['sentiment_logits'], dim=1).cpu().numpy()
                theme_preds = torch.argmax(outputs['theme_logits'], dim=1).cpu().numpy()
                
                all_sentiment_preds.extend(sentiment_preds)
                all_sentiment_true.extend(batch['sentiment_label'].cpu().numpy())
                all_theme_preds.extend(theme_preds)
                all_theme_true.extend(batch['theme_label'].cpu().numpy())
        
        # Average losses and metrics
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        # Generate detailed classification reports
        if self.current_epoch % 10 == 0:  # Every 10 epochs
            self._log_detailed_metrics(all_sentiment_preds, all_sentiment_true, 
                                     all_theme_preds, all_theme_true)
        
        return {**avg_losses, **avg_metrics}
    
    def _log_detailed_metrics(self, sentiment_preds, sentiment_true, theme_preds, theme_true):
        """Log detailed classification metrics."""
        # Sentiment classification report
        sentiment_report = classification_report(
            sentiment_true, sentiment_preds,
            target_names=list(SentimentThemeConfig.SENTIMENT_LABELS.values()),
            output_dict=True
        )
        
        # Theme classification report
        theme_labels = list(SentimentThemeConfig.get_theme_labels(
            self.train_loader.dataset.domain
        ).values())
        theme_report = classification_report(
            theme_true, theme_preds,
            target_names=theme_labels,
            output_dict=True
        )
        
        if self.use_wandb:
            # Log classification reports to wandb
            wandb.log({
                'val/sentiment_precision': sentiment_report['weighted avg']['precision'],
                'val/sentiment_recall': sentiment_report['weighted avg']['recall'],
                'val/theme_precision': theme_report['weighted avg']['precision'],
                'val/theme_recall': theme_report['weighted avg']['recall'],
                'epoch': self.current_epoch
            })
            
            # Log confusion matrices
            self._log_confusion_matrices(sentiment_preds, sentiment_true, 
                                       theme_preds, theme_true)
    
    def _log_confusion_matrices(self, sentiment_preds, sentiment_true, theme_preds, theme_true):
        """Log confusion matrices to wandb."""
        # Sentiment confusion matrix
        sentiment_cm = confusion_matrix(sentiment_true, sentiment_preds)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.heatmap(sentiment_cm, annot=True, fmt='d', ax=ax1,
                   xticklabels=list(SentimentThemeConfig.SENTIMENT_LABELS.values()),
                   yticklabels=list(SentimentThemeConfig.SENTIMENT_LABELS.values()))
        ax1.set_title('Sentiment Confusion Matrix')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Theme confusion matrix
        theme_cm = confusion_matrix(theme_true, theme_preds)
        theme_labels = list(SentimentThemeConfig.get_theme_labels(
            self.train_loader.dataset.domain
        ).values())
        
        sns.heatmap(theme_cm, annot=True, fmt='d', ax=ax2,
                   xticklabels=theme_labels,
                   yticklabels=theme_labels)
        ax2.set_title('Theme Confusion Matrix')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if self.use_wandb:
            wandb.log({"confusion_matrices": wandb.Image(fig)})
        
        plt.close()
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model with validation accuracy: {metrics['overall_acc']:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, epochs: int, resume_from: Optional[str] = None):
        """Main training loop."""
        if resume_from:
            self.load_checkpoint(resume_from)
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['total'])
            self.train_metrics.append(train_metrics)
            
            # Validation phase
            val_metrics = self.validate_epoch()
            self.val_losses.append(val_metrics['total'])
            self.val_metrics.append(val_metrics)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total'])
                else:
                    self.scheduler.step()
            
            # Check for best model
            is_best = val_metrics['overall_acc'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['overall_acc']
                self.best_val_loss = val_metrics['total']
            
            # Save checkpoint
            if epoch % 10 == 0 or is_best:
                self.save_checkpoint(val_metrics, is_best)
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_metrics['total'],
                    'train/sentiment_acc': train_metrics['sentiment_acc'],
                    'train/theme_acc': train_metrics['theme_acc'],
                    'train/overall_acc': train_metrics['overall_acc'],
                    'val/loss': val_metrics['total'],
                    'val/sentiment_acc': val_metrics['sentiment_acc'],
                    'val/theme_acc': val_metrics['theme_acc'],
                    'val/overall_acc': val_metrics['overall_acc'],
                    'val/quality_mae': val_metrics['quality_mae'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{epochs-1}")
            print(f"Train Loss: {train_metrics['total']:.4f}, Train Acc: {train_metrics['overall_acc']:.4f}")
            print(f"Val Loss: {val_metrics['total']:.4f}, Val Acc: {val_metrics['overall_acc']:.4f}")
            print(f"Best Val Acc: {self.best_val_acc:.4f}")
            print("-" * 50)
        
        print("Training completed!")
        
        # Generate final evaluation report
        self._generate_final_report()
    
    def _generate_final_report(self):
        """Generate comprehensive training report."""
        # Plot training curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(len(self.train_losses))
        
        # Loss curves
        axes[0, 0].plot(epochs, self.train_losses, label='Train')
        axes[0, 0].plot(epochs, self.val_losses, label='Validation')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        train_accs = [m['overall_acc'] for m in self.train_metrics]
        val_accs = [m['overall_acc'] for m in self.val_metrics]
        
        axes[0, 1].plot(epochs, train_accs, label='Train')
        axes[0, 1].plot(epochs, val_accs, label='Validation')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Sentiment accuracy
        train_sent_accs = [m['sentiment_acc'] for m in self.train_metrics]
        val_sent_accs = [m['sentiment_acc'] for m in self.val_metrics]
        
        axes[1, 0].plot(epochs, train_sent_accs, label='Train')
        axes[1, 0].plot(epochs, val_sent_accs, label='Validation')
        axes[1, 0].set_title('Sentiment Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Theme accuracy
        train_theme_accs = [m['theme_acc'] for m in self.train_metrics]
        val_theme_accs = [m['theme_acc'] for m in self.val_metrics]
        
        axes[1, 1].plot(epochs, train_theme_accs, label='Train')
        axes[1, 1].plot(epochs, val_theme_accs, label='Validation')
        axes[1, 1].set_title('Theme Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.checkpoint_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        if self.use_wandb:
            wandb.log({"training_curves": wandb.Image(str(plot_path))})
        
        plt.close()
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training report saved to {self.checkpoint_dir}")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


def create_training_config(domain: str = "fashion") -> Dict[str, Any]:
    """Create default training configuration."""
    return {
        # Model parameters
        'clip_model_name': 'openai/clip-vit-base-patch32',
        'hidden_dim': 512,
        'dropout_rate': 0.1,
        
        # Training parameters
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'clip_learning_rate': 1e-5,
        'head_learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        
        # Loss weights
        'sentiment_weight': 1.0,
        'theme_weight': 1.0,
        'quality_weight': 0.5,
        
        # Technical parameters
        'use_amp': True,
        'num_workers': 4,
        'pin_memory': True,
        
        # Checkpointing
        'checkpoint_dir': f'checkpoints/{domain}_sentiment_analysis',
        'save_every': 10,
        
        # Domain specific
        'domain': domain,
        'num_sentiment_classes': 5,
        'num_theme_classes': 10 if domain == 'fashion' else 6,
    }