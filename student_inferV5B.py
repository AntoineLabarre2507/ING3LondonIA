import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.transforms import functional as F
from collections import Counter

class EnsembleModel:
    def __init__(self, models_list, device):
        self.models = models_list
        self.device = device

    def eval(self):
        for m in self.models: m.eval()

    def to(self, device):
        for m in self.models: m.to(device)
        self.device = device
        return self

def build_model(torch, nn, models, classes=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_list = []
    final_classes = None

    # 1. ResNet-101 V3
    try:
        ckpt = torch.load("resnet101_best_v3.pt", map_location="cpu")
        final_classes = [str(c) for c in ckpt["classes"]]
        m = models.resnet101(weights=None)
        m.fc = nn.Linear(m.fc.in_features, len(final_classes))
        m.load_state_dict(ckpt["model"], strict=True)
        m.to(device).eval()
        models_list.append(m)
        print("✅ ResNet-101 chargé")
    except: print("⚠️ ResNet-101 non chargé")

    # 2. EfficientNet-B3
    try:
        ckpt = torch.load("efficientnet_b3_best.pt", map_location="cpu")
        cls_b3 = [str(c) for c in ckpt["classes"]]
        m = models.efficientnet_b3(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(cls_b3))
        m.load_state_dict(ckpt["model"], strict=True)
        m.to(device).eval()
        models_list.append(m)
        print("✅ EfficientNet-B3 chargé")
    except: print("⚠️ EfficientNet-B3 non chargé")

    # 3. DenseNet-121
    try:
        ckpt = torch.load("densenet121_best.pt", map_location="cpu")
        cls_dn = [str(c) for c in ckpt["classes"]]
        m = models.densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, len(cls_dn))
        m.load_state_dict(ckpt["model"], strict=True)
        m.to(device).eval()
        models_list.append(m)
        print("✅ DenseNet-121 chargé")
    except: print("⚠️ DenseNet-121 non chargé")

    # 4. Swin-T
    try:
        ckpt = torch.load("swin_t_best.pt", map_location="cpu")
        cls_sw = [str(c) for c in ckpt["classes"]]
        m = models.swin_t(weights=None)
        m.head = nn.Linear(m.head.in_features, len(cls_sw))
        m.load_state_dict(ckpt["model"], strict=True)
        m.to(device).eval()
        models_list.append(m)
        print("✅ Swin-T chargé")
    except: print("⚠️ Swin-T non chargé")

    if not models_list: raise RuntimeError("❌ Aucun modèle chargé !")
    return EnsembleModel(models_list, device), final_classes

def predict(ensemble, image, preprocess, torch):
    """
    Prédiction par VOTING PONDÉRÉ au lieu de moyenne de probabilités
    Souvent plus robuste pour l'ensembling
    """
    if image.mode != "RGB": image = image.convert("RGB")
    device = ensemble.device
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    # TTA avec 5 vues
    tta_transforms = [
        transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm]),
        transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), norm]),
        transforms.Compose([transforms.Resize(288), transforms.CenterCrop(256), transforms.Resize(224), transforms.ToTensor(), norm]),
        transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), norm]),
        transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ColorJitter(brightness=0.1, contrast=0.1), transforms.ToTensor(), norm]),
    ]
    
    # Poids pour le voting
    if len(ensemble.models) == 4:
        model_weights = [3, 3, 2, 2]  # ResNet, EfficientNet, DenseNet, Swin-T
    else:
        model_weights = [6, 2, 2]  # Sans Swin-T
    
    votes = []
    
    with torch.inference_mode():
        for i, model in enumerate(ensemble.models):
            for tf in tta_transforms:
                try:
                    img_tensor = tf(image).unsqueeze(0).to(device)
                    logits = model(img_tensor)
                    pred = logits.argmax(dim=1).item()
                    
                    # Ajouter le vote avec répétition selon le poids
                    votes.extend([pred] * model_weights[i])
                except:
                    continue
    
    # Vote majoritaire pondéré
    if votes:
        vote_counts = Counter(votes)
        final_prediction = vote_counts.most_common(1)[0][0]
        return final_prediction
    
    # Fallback : si aucun vote, utiliser la moyenne classique
    return 0
