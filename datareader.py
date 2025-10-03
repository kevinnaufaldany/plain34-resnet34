import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
RANDOM_SEED = 110
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Dataset configuration
TRAIN_CSV = "train2.csv"
TRAIN_DIR = "train2"

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Augmentasi untuk training
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Transformasi untuk validasi (tanpa augmentasi)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def create_train_val_split(csv_file, train_ratio=0.8, random_state=42):
    """
    Membagi data menjadi train dan validation secara stratified.
    """
    df = pd.read_csv(csv_file)
    train_df, val_df = train_test_split(
        df,
        test_size=1-train_ratio,
        random_state=random_state,
        stratify=df['label']
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


class FoodDataset(Dataset):
    """
    PyTorch Dataset untuk data makanan Indonesia.
    """
    def __init__(self, dataframe, img_dir, label2idx, img_size=224, transform=None, infer=False):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.label2idx = label2idx
        self.img_size = (img_size, img_size)
        self.transform = transform
        self.infer = infer
        self.labels = list(label2idx.keys())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = str(self.df.iloc[idx]['filename'])
        img_path = os.path.join(self.img_dir, filename)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', self.img_size, color='gray')
        if self.transform:
            image = self.transform(image)
        if self.infer:
            return image, filename
        label_name = self.df.iloc[idx]['label']
        label = self.label2idx[label_name]
        return image, label


def get_label_mapping(labels):
    """
    Membuat mapping label ke index sesuai urutan yang diinginkan.
    """
    desired_order = ['nasi_goreng', 'rendang', 'soto_ayam', 'bakso', 'gado_gado']
    return {label: idx for idx, label in enumerate(desired_order) if label in labels}


def prepare_datasets(csv_path=TRAIN_CSV, img_dir=TRAIN_DIR, img_size=224, train_ratio=0.8):
    """
    Fungsi utama untuk menyiapkan dataset train dan validasi.
    """
    train_df, val_df = create_train_val_split(csv_path, train_ratio=train_ratio, random_state=RANDOM_SEED)
    labels = sorted(train_df['label'].unique())
    label2idx = get_label_mapping(labels)

    train_ds = FoodDataset(train_df, img_dir, label2idx, img_size, transform=train_transform)
    val_ds = FoodDataset(val_df, img_dir, label2idx, img_size, transform=val_transform)

    print(f"Train samples: {len(train_ds)} | Validation samples: {len(val_ds)}")
    print(f"Label mapping: {label2idx}")
    return train_ds, val_ds, label2idx
