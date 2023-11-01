from model import CNN
from train_eval import train, eval
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from dataset import FaceDataset
import os
import torch
import random
import numpy as np


if __name__ == "__main__":
    # 获取当前脚本所在的绝对路径
    current_script_path = os.path.abspath(__file__)
    # 获取当前脚本所在的目录
    current_script_directory = os.path.dirname(current_script_path)
    # 设置当前工作目录为脚本所在的目录
    os.chdir(current_script_directory)

    # 设置随机数种子
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None
    random.seed(seed)
    np.random.seed(seed)

    test_dir = r"./dataset/test"
    train_dir = r"./dataset/train"
    val_dir = r"./dataset/valid"
    batch_size = 32

    transform = transforms.Compose([transforms.ToTensor()])

    # 创建数据集和数据加载器
    train_dataset = FaceDataset(train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = FaceDataset(val_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataset = FaceDataset(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = CNN(num_classes=3)
    model = train(model, train_loader, val_loader, num_epochs=10, patience=3)
    eval(model, test_loader)
