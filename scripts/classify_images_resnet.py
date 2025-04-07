import os
import torch
from torchvision import models, transforms
from PIL import Image
import shutil

def load_model(model_path, num_classes=3):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def predict_image(model, image_path, transform, class_names):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1).item()
    return class_names[pred]

def classify_images(input_dir, model_path, output_dir, class_names):
    os.makedirs(output_dir, exist_ok=True)
    for cname in class_names:
        os.makedirs(os.path.join(output_dir, cname), exist_ok=True)

    model = load_model(model_path, num_classes=len(class_names))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    for fname in os.listdir(input_dir):
        fpath = os.path.join(input_dir, fname)
        if not fpath.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        pred_class = predict_image(model, fpath, transform, class_names)
        print(f"[INFO] {fname} → {pred_class}")
        shutil.copy(fpath, os.path.join(output_dir, pred_class, fname))

if __name__ == "__main__":
    input_dir = "pdf-ocr-dl/data/outputs/images"
    model_path = "pdf-ocr-dl/models/image_classifier_resnet18.pth"
    output_dir = "pdf-ocr-dl/data/outputs/categorized_images"
    class_names = ["chart", "figure", "logo"]

    classify_images(input_dir, model_path, output_dir, class_names)
