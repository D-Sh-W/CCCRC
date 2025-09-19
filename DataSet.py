import os
import csv
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


def enhance_vertical_lines(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(img)
    sobel_y = cv2.Sobel(contrast_enhanced, cv2.CV_64F, dx=0, dy=1, ksize=3)
    sobel_y = np.uint8(np.absolute(sobel_y))
    enhanced_img = cv2.addWeighted(contrast_enhanced, 0.7, sobel_y, 0.3, 0)
    return enhanced_img


class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.data.append((row[0], int(row[1])))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        folder_path, label = self.data[idx]
        root_dir = "./test_data/"
        img_path1 = root_dir + os.path.join(folder_path, "dm_time.png")
        img_path2 = root_dir + os.path.join(folder_path, "freq_time.png")
        enhanced_img = enhance_vertical_lines(img_path2)
        image1 = Image.open(img_path1).convert("RGB")
        image2 = Image.fromarray(enhanced_img).convert("RGB")
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, torch.tensor(label, dtype=torch.long), folder_path
