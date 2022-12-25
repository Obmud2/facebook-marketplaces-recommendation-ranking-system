import os
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

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
    optimiser = torch.optim.SGD(model.parameters(), lr=0.1)
    for epoch in range(epochs):
        for batch in train_loader:
            features, labels = batch
            prediction = model(features)
            loss = F.cross_entropy(prediction, labels.long())
            loss.backward()
            print(loss.item())
            optimiser.step()
            optimiser.zero_grad()

if __name__ == "__main__":
    dataset = ImageDataset()
    print(type(dataset[0][1]))
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = CNN()
    train(model)