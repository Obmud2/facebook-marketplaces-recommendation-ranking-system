import os, math
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.models import resnet50, ResNet50_Weights
import torchvision
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

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

class TransferLearning(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        for i, param in enumerate(self.layers.parameters()):
            if i<158:
                param.requires_grad = False
            print(f"{i} {param.requires_grad}: {param.shape}")

        self.layers.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 13)
        )
        
    def forward(self, features):
        return self.layers(features)

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 3),
            torch.nn.Flatten(),
            torch.nn.Linear(30976, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 13),
            torch.nn.Softmax()
        )

    def forward(self, features):
        return self.layers(features)

def train(model, epochs=50):
    run_time = datetime.now().replace(microsecond=0).isoformat()
    folder_name = f"model_evaluation/{run_time}/weights"
    os.makedirs(folder_name)
    model.train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=5, gamma=0.1)
    writer = SummaryWriter()
    batch_idx = 0
    for epoch in range(epochs):
        total_correct = 0
        total = 0
        avg_loss = []
        print(f"Epoch: {epoch+1}/{epochs} \tLR: {scheduler.get_last_lr()[0]}")
        for batch in tqdm(train_loader, desc=f"Training: "):
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            prediction = model(features)

            loss = F.cross_entropy(prediction, labels.long())
            loss.backward()
            avg_loss.append(loss.item())
            optimiser.step()
            optimiser.zero_grad()

            prediction_values = prediction.max(1)[1]
            no_batch_values = prediction_values.size(dim=0)
            no_correct = sum(prediction_values == labels)
            batch_acc = no_correct / no_batch_values
            total_correct += no_correct
            total += no_batch_values
            writer.add_scalar("Loss", loss.item(), batch_idx)
            writer.add_scalar("Acc", batch_acc, batch_idx)
            batch_idx += 1

        epoch_acc, epoch_loss = (total_correct/len(train_loader.dataset), sum(avg_loss)/len(avg_loss))
        valid_acc, valid_loss = validate_model(model, valid_loader)
        writer.add_scalars(f'Loss (per epoch)', {'loss': epoch_loss, 'valid_loss': valid_loss}, epoch)
        print(f"Loss: {'%.5f'%epoch_loss} \tAcc: {('%.3f'%(epoch_acc*100)).rjust(6)}% \tValidation Loss: {'%.5f'%valid_loss}\tValidation Acc: {('%.3f'%(valid_acc*100)).rjust(6)}%")
        torch.save(model.state_dict(), os.path.join(folder_name, f"epoch_{epoch}.pt"))
        model_data = {
            "model": model.__class__.__name__,
            "date_trained": run_time,
            "run_finished": datetime.now().replace(microsecond=0).isoformat(),
            "epoch": epoch,
            "train_acc": epoch_acc.item(),
            "train_loss": epoch_loss,
            "valid_acc": valid_acc.item(),
            "valid_loss": valid_loss
        }
        with open(os.path.join(folder_name, f'epoch_{epoch}_data.json'), 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=4)
        # scheduler.step()
    return model_data

def validate_model(model, dataloader):
    #model.eval()
    total_correct = 0
    loss = 0
    with torch.no_grad():
        for batch in dataloader:
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            prediction = model(features)

            prediction_values = prediction.max(1)[1]
            total_correct += sum(prediction_values == labels)
            loss += F.cross_entropy(prediction, labels.long()).item() * len(prediction_values)

        acc = total_correct / len(dataloader.dataset)
        loss = loss / len(dataloader.dataset)
    return (acc, loss)

def select_device():
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device

if __name__ == "__main__":
    device = select_device()
    print(f"Using device: {device.type}")
    dataset = ImageDataset()
    batch_size = 32
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [0.7, 0.15, 0.15], generator=torch.Generator().manual_seed(41))
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size)
    valid_loader = DataLoader(validation_dataset, batch_size)
    model = TransferLearning().to(device)
    model_data = train(model)

    torch.save(model.state_dict(), 'final_models/image_model.pt')
    with open('final_models/model_data.json', 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=4)

    """
    # Load model:
    state_dict = torch.load('model.pt')
    new_model = CNN().to(device)
    new_model.load_state_dict(state_dict)
    """
