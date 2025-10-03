
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import numpy as np

from resnet34_improved import create_resnet34_improved
from datareader import prepare_datasets
from utils import check_set_gpu


# ============ Configuration ============
CONFIG = {
    'batch_size': 32,           # Increased for better gradient estimates
    'epochs': 100,              # More epochs with early stopping
    'learning_rate': 3e-4,      # Lower LR for stability
    'weight_decay': 1e-4,       # L2 regularization
    'label_smoothing': 0.1,     # Label smoothing untuk generalisasi
    'dropout_rate': 0.2,        # Dropout dalam model
    'use_se': True,             # SE blocks
    'patience': 15,             # Early stopping patience
    'grad_clip': 1.0,           # Gradient clipping
    'warmup_epochs': 5,         # LR warmup
}

DEVICE = check_set_gpu()
# print(f"Using device: {DEVICE}")
# print("="*60)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    Mencegah overconfidence dan meningkatkan generalisasi
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)
        
        # One-hot encoding with smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        return torch.mean(torch.sum(-true_dist * log_preds, dim=-1))


class EarlyStopping:
    """Early stopping untuk mencegah overfitting"""
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_acc):
        score = val_acc
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine annealing dengan warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    """Training untuk satu epoch dengan mixed precision"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch+1} [Train]', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward dengan gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        current_acc = correct / total
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.4f}'})
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device, epoch):
    """Validasi"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f'Epoch {epoch+1} [Val]  ', leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            current_acc = correct / total
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.4f}'})
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def plot_training_curves(history, save_path='resnet3'):
    """Plot dan simpan training curves"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/training_curves_improved.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {save_path}/training_curves_improved.png")


def main():
    """Main training loop"""
    print("="*60)
    print("ResNet-34 Improved Training")
    print("="*60)
    print("Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("="*60)
    
    # Load datasets
    print("\nLoading datasets...")
    train_ds, val_ds, label2idx = prepare_datasets()
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print(f"Label mapping: {label2idx}")
    print("="*60)
    
    # Create model
    print("\nCreating model...")
    num_classes = len(label2idx)
    model = create_resnet34_improved(
        num_classes=num_classes,
        dropout_rate=CONFIG['dropout_rate'],
        use_se=CONFIG['use_se']
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print("="*60)
    
    # Loss and optimizer
    criterion = LabelSmoothingCrossEntropy(smoothing=CONFIG['label_smoothing'])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    num_training_steps = CONFIG['epochs'] * len(train_loader)
    num_warmup_steps = CONFIG['warmup_epochs'] * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Early stopping
    early_stopping = EarlyStopping(patience=CONFIG['patience'])
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    
    print("\nStarting training...")
    print("="*60)
    
    for epoch in range(CONFIG['epochs']):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, DEVICE, epoch
        )
        
        # Update scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': CONFIG
            }, 'resnet3/best_resnet34_improved.pth')
            print(f"  → Model saved! (Best Val Acc: {best_val_acc:.4f})")
        
        # Early stopping check
        early_stopping(val_acc)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    print("="*60)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("="*60)
    
    # Save history
    with open('resnet3/training_history_improved.json', 'w') as f:
        json.dump(history, f, indent=4)
    print("Training history saved to resnet3/training_history_improved.json")
    
    # Plot curves
    plot_training_curves(history, save_path='resnet3')
    
    print("\nAll files saved successfully!")


if __name__ == "__main__":
    import torch.nn.functional as F  # Import here to avoid circular import
    main()
