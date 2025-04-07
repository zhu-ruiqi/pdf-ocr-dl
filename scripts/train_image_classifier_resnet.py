import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report
import numpy as np

def train_image_classifier(data_dir, epochs=10, batch_size=4, num_classes=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_dataset = ImageFolder(train_dir, transform=transform)
    val_dataset = ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f"[INFO] Starting training on {device}...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[EPOCH {epoch+1}] Loss: {total_loss / len(train_loader):.4f}")

    # Evaluation
    print("[INFO] Evaluating on validation set...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\n[RESULT] Classification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=val_dataset.classes))

    # Save model
    model_path = os.path.join("pdf-ocr-dl", "models", "image_classifier_resnet18.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Model saved as {model_path}")

if __name__ == "__main__":
    data_dir = "pdf-ocr-dl/data/image_classification"
    train_image_classifier(data_dir)
