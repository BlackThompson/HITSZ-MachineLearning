import os
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class FaceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = os.listdir(data_dir)
        self.label_encoder = LabelEncoder()

        self.labels = [file.split("_")[-1] for file in self.image_files]
        self.labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(image_name)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
