import os, math
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

class ImageDataset(Dataset):
    def __init__(self, annotations_file="data/labels.csv", img_dir="data/clean_images/"):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img = Image.open(img_path)
        img = torchvision.transforms.ToTensor()(img)
        label = self.img_labels.iloc[idx, 1]
        return img, label

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 3),
            torch.nn.Flatten(),
            torch.nn.Linear(25600, 2000),
            torch.nn.ReLU(),
            torch.nn.Linear(2000, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 13),
            torch.nn.Softmax()
        )

    def forward(self, features):
        return self.layers(features)

def train(model, epochs=10):
    model.train()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.5)
    writer = SummaryWriter()
    batch_idx = 0
    for epoch in range(epochs):
        for batch in train_loader:
            features, labels = batch
            prediction = model(features)
            prediction_values = prediction.max(1)[1]
            acc = int(sum(prediction_values == labels)) / len(prediction_values)
            loss = F.cross_entropy(prediction, labels.long())
            loss.backward()
            print(f"Epoch: {epoch}\tLoss: {'%.5f'%loss.item()}\tAcc: {('%.3f'%(acc*100)).rjust(6)}%")
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar("Loss", loss.item(), batch_idx)
            writer.add_scalar("Acc", acc, batch_idx)
            batch_idx += 1

if __name__ == "__main__":
    dataset = ImageDataset()
    batch_size = 64
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [0.7, 0.15, 0.15], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size)
    valid_loader = DataLoader(validation_dataset, batch_size)
    model = CNN()
    train(model)