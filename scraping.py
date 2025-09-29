import os
import csv
import shutil
from collections import defaultdict

# Konfigurasi
input_csv = 'train.csv'
output_csv = 'train2.csv'
src_folder = 'train'
dst_folder = 'train2'
max_per_class = 100

# Membuat folder tujuan jika belum ada
os.makedirs(dst_folder, exist_ok=True)

# Membaca data dari train.csv dan memilih maksimal 100 per kelas
data_per_class = defaultdict(list)
with open(input_csv, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        label = row['label']
        if len(data_per_class[label]) < max_per_class:
            data_per_class[label].append(row)

# Menyalin file gambar terpilih ke folder baru dan menulis CSV baru
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['filename', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for label, items in data_per_class.items():
        for item in items:
            src_path = os.path.join(src_folder, item['filename'])
            dst_path = os.path.join(dst_folder, item['filename'])
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
            writer.writerow(item)

print(f'Selesai! File {output_csv} dan folder {dst_folder} berisi maksimal {max_per_class} data per kelas.')
