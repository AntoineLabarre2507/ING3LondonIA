import torch
import torch.nn as nn
from torchvision import models, transforms

def build_model(torch, nn, models, classes):
    # Assure-toi que c'est bien le bon fichier V3
    ckpt = torch.load("resnet101_best_v3.pt", map_location="cpu")
    classes = [str(c) for c in ckpt["classes"]]
    
    model = models.resnet101(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    
    return model, classes

def predict(model, image, preprocess, torch):
    """
    Inférence avec TTA simple (Moyenne de l'image originale et retournée).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    device = next(model.parameters()).device
    
    # Définition manuelle des transforms pour être sûr de ce qu'on fait
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    # 1. Vue normale
    t1 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        norm
    ])
    
    # 2. Vue miroir (flip horizontal forcé)
    t2 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        norm
    ])

    # Création du batch de 2 images [Originale, Miroir]
    batch = torch.stack([t1(image), t2(image)]).to(device)

    with torch.inference_mode():
        logits = model(batch)
        probs = torch.nn.functional.softmax(logits, dim=1)
        # Moyenne des probabilités des 2 vues
        avg_probs = probs.mean(dim=0)
        pred_idx = int(avg_probs.argmax().item())

    return pred_idx