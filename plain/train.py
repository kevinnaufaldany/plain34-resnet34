import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datareader import prepare_datasets
from plain34 import create_plain34
from utils import check_set_gpu
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

# Konfigurasi
BATCH_SIZE = 8
EPOCHS = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2   # Tambahkan weight decay

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
	"""
	Fungsi utama training dan validasi model Plain34.
	"""
	# Load dataset dan split dengan augmentasi tambahan
	train_ds, val_ds, label2idx = prepare_datasets(
		# augment=True  # Pastikan di datareader kamu tambahkan augmentasi extra
	)
	train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
	val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

	# Load model
	num_classes = len(label2idx)
	model = create_plain34(num_classes=num_classes).to(DEVICE)

	# Loss dan optimizer (AdamW + weight decay)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

	# History dict
	history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

	def train_one_epoch(model, loader, criterion, optimizer, device):
		model.train()
		running_loss, correct, total = 0.0, 0, 0
		pbar = tqdm(loader, desc='Train', leave=False)
		for images, labels in pbar:
			images, labels = images.to(device), labels.to(device)
			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item() * images.size(0)
			_, predicted = outputs.max(1)
			correct += predicted.eq(labels).sum().item()
			total += labels.size(0)
			pbar.set_postfix({'loss': f'{loss.item():.4f}'})
		return running_loss / total, correct / total

	def validate(model, loader, criterion, device):
		model.eval()
		running_loss, correct, total = 0.0, 0, 0
		with torch.no_grad():
			pbar = tqdm(loader, desc='Val', leave=False)
			for images, labels in pbar:
				images, labels = images.to(device), labels.to(device)
				outputs = model(images)
				loss = criterion(outputs, labels)
				running_loss += loss.item() * images.size(0)
				_, predicted = outputs.max(1)
				correct += predicted.eq(labels).sum().item()
				total += labels.size(0)
				pbar.set_postfix({'loss': f'{loss.item():.4f}'})
		return running_loss / total, correct / total

	best_val_acc = 0.0
	for epoch in range(EPOCHS):
		train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
		val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

		# Simpan ke history
		history["train_loss"].append(train_loss)
		history["train_acc"].append(train_acc)
		history["val_loss"].append(val_loss)
		history["val_acc"].append(val_acc)

		print(f"Epoch {epoch+1}/{EPOCHS} | "
		      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
		      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			torch.save(model.state_dict(), "best_plain34.pth")
			print("Model saved!")

	# Simpan history ke file JSON
	with open("training_history.json", "w") as f:
		json.dump(history, f)

	# Visualisasi Loss
	plt.figure(figsize=(10, 5))
	plt.plot(history["train_loss"], label="Train Loss")
	plt.plot(history["val_loss"], label="Val Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("Training & Validation Loss Plain34")
	plt.legend()
	plt.savefig("loss_curve_plain34.png")
	plt.close()

	# Visualisasi Accuracy
	plt.figure(figsize=(10, 5))
	plt.plot(history["train_acc"], label="Train Acc")
	plt.plot(history["val_acc"], label="Val Acc")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.title("Training & Validation Accuracy Plain34")
	plt.legend()
	plt.savefig("accuracy_curve_plain34.png")
	plt.close()


if __name__ == "__main__":
	main()
