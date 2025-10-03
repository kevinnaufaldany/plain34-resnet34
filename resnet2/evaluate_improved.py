import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json

from resnet34_improved import create_resnet34_improved
from datareader import prepare_datasets
from utils import check_set_gpu


DEVICE = check_set_gpu()


def load_model(checkpoint_path, num_classes=5):
    """Load best model from checkpoint"""
    model = create_resnet34_improved(num_classes=num_classes, dropout_rate=0.2, use_se=True)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Best val accuracy: {checkpoint['val_acc']:.4f}")
    print(f"Best val loss: {checkpoint['val_loss']:.4f}")
    
    return model, checkpoint


def evaluate_model(model, val_loader, device):
    """Comprehensive evaluation"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='resnet3'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - ResNet-34 Improved', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{save_path}/confusion_matrix_improved.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}/confusion_matrix_improved.png")


def print_classification_report(y_true, y_pred, class_names):
    """Print detailed classification report"""
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(report)
    
    # Save to file
    with open('resnet3/classification_report_improved.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("CLASSIFICATION REPORT - ResNet-34 Improved\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    
    print("Report saved to resnet3/classification_report_improved.txt")


def analyze_per_class_accuracy(y_true, y_pred, class_names, save_path='resnet3'):
    """Analyze and plot per-class accuracy"""
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, per_class_acc, color='skyblue', edgecolor='navy', linewidth=1.5)
    
    # Add value labels on bars
    for bar, acc in zip(bars, per_class_acc):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Per-Class Accuracy - ResNet-34 Improved', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}/per_class_accuracy_improved.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Per-class accuracy plot saved to {save_path}/per_class_accuracy_improved.png")
    
    # Print results
    print("\n" + "="*60)
    print("PER-CLASS ACCURACY")
    print("="*60)
    for class_name, acc in zip(class_names, per_class_acc):
        print(f"{class_name:15s}: {acc:.2%}")


def compare_with_baseline(save_path='resnet3'):
    """Compare with baseline and standard ResNet-34"""
    # Load history
    try:
        with open(f'{save_path}/training_history_improved.json', 'r') as f:
            history_improved = json.load(f)
        
        # Get best metrics
        best_val_acc_improved = max(history_improved['val_acc'])
        best_val_loss_improved = min(history_improved['val_loss'])
        final_train_acc_improved = history_improved['train_acc'][-1]
        
        print("\n" + "="*60)
        print("COMPARISON WITH BASELINE")
        print("="*60)
        print("\nResNet-34 Improved (Current):")
        print(f"  Best Val Accuracy:  {best_val_acc_improved:.4f}")
        print(f"  Best Val Loss:      {best_val_loss_improved:.4f}")
        print(f"  Final Train Acc:    {final_train_acc_improved:.4f}")
        print(f"  Total Epochs:       {len(history_improved['train_loss'])}")
        
        # Try to load baseline for comparison
        baseline_paths = [
            '../plain/training_history_plain34.json',
            'training_history_modified.json'
        ]
        
        for baseline_path in baseline_paths:
            full_path = os.path.join(save_path, baseline_path)
            if os.path.exists(full_path):
                with open(full_path, 'r') as f:
                    history_baseline = json.load(f)
                
                best_val_acc_baseline = max(history_baseline.get('val_acc', [0]))
                improvement = (best_val_acc_improved - best_val_acc_baseline) * 100
                
                print(f"\nBaseline ({baseline_path}):")
                print(f"  Best Val Accuracy:  {best_val_acc_baseline:.4f}")
                print(f"\nImprovement:")
                print(f"  Δ Accuracy:         +{improvement:.2f}%")
                break
        
    except FileNotFoundError:
        print("Training history not found. Please train the model first.")


def main():
    """Main evaluation script"""
    print("="*60)
    print("ResNet-34 Improved - Evaluation")
    print("="*60)
    
    # Load datasets
    print("\nLoading datasets...")
    _, val_ds, label2idx = prepare_datasets()
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
    
    # Get class names
    idx2label = {v: k for k, v in label2idx.items()}
    class_names = [idx2label[i] for i in range(len(label2idx))]
    
    print(f"Validation samples: {len(val_ds)}")
    print(f"Classes: {class_names}")
    
    # Load model
    print("\nLoading model...")
    checkpoint_path = 'resnet3/best_resnet34_improved.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train_improved.py")
        return
    
    model, checkpoint = load_model(checkpoint_path, num_classes=len(label2idx))
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred, y_true, y_probs = evaluate_model(model, val_loader, DEVICE)
    
    # Calculate overall accuracy
    accuracy = (y_pred == y_true).mean()
    print(f"\nOverall Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Generate visualizations and reports
    print("\nGenerating reports and visualizations...")
    plot_confusion_matrix(y_true, y_pred, class_names)
    print_classification_report(y_true, y_pred, class_names)
    analyze_per_class_accuracy(y_true, y_pred, class_names)
    compare_with_baseline()
    
    print("\n" + "="*60)
    print("Evaluation completed successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  - resnet3/confusion_matrix_improved.png")
    print("  - resnet3/classification_report_improved.txt")
    print("  - resnet3/per_class_accuracy_improved.png")


if __name__ == "__main__":
    main()
