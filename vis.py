import torch
import matplotlib.pyplot as plt
from plain34 import create_plain34
from datareader import prepare_datasets

# Konfigurasi
BATCH_SIZE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
train_ds, val_ds, label2idx = prepare_datasets()
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# Load model dan weight
num_classes = len(label2idx)
model = create_plain34(num_classes=num_classes).to(DEVICE)
model.load_state_dict(torch.load('best_plain34.pth', map_location=DEVICE))
model.eval()

# Evaluasi di validation set
val_loss = 0.0
val_correct = 0
val_total = 0
criterion = torch.nn.CrossEntropyLoss()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        val_correct += predicted.eq(labels).sum().item()
        val_total += labels.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
val_loss = val_loss / val_total
val_acc = val_correct / val_total

print(f"Baseline Plain34 - Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.4f}")

# Visualisasi (dummy history, ganti dengan history asli jika ada)
history = {
    'train_loss': [1.5, 1.2, 1.0, 0.8, 0.7],      # ganti dengan data asli jika ada
    'val_loss':   [1.6, 1.3, 1.1, 0.9, 0.8],      # ganti dengan data asli jika ada
    'train_acc':  [0.25, 0.40, 0.55, 0.65, 0.72], # ganti dengan data asli jika ada
    'val_acc':    [0.28, 0.38, 0.50, 0.60, 0.68], # ganti dengan data asli jika ada
}
epochs = range(1, len(history['train_loss']) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, history['train_loss'], label='Train Loss')
plt.plot(epochs, history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Training & Validasi')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, history['train_acc'], label='Train Acc')
plt.plot(epochs, history['val_acc'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Akurasi Training & Validasi')
plt.legend()

plt.tight_layout()
plt.savefig('plain34_baseline_performance.png')
plt.show()

