import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.transforms import functional as F

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
    if image.mode != "RGB": image = image.convert("RGB")
    device = ensemble.device
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    base_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm])
    
    # TTA 2 vues
    batch = torch.stack([base_tf(image), base_tf(F.hflip(image))]).to(device)

    total_probs = None
    with torch.inference_mode():
        for model in ensemble.models:
            logits = model(batch)
            probs = torch.nn.functional.softmax(logits, dim=1).mean(dim=0)
            if total_probs is None: total_probs = probs
            else: total_probs += probs

    final_probs = total_probs / len(ensemble.models)
    return int(final_probs.argmax().item())