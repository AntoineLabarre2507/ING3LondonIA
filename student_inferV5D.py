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
    FUSION HYBRIDE : Moyenne pondérée + Voting
    - Si la confiance moyenne est haute (>0.7) : utiliser moyenne pondérée
    - Sinon : utiliser voting pour plus de robustesse
    """
    if image.mode != "RGB": image = image.convert("RGB")
    device = ensemble.device
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    # TTA riche
    base_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    
    augmentations = [
        base_tf(image),  # Original
        base_tf(F.hflip(image)),  # Flip H
        F.center_crop(F.resize(image, 288), 224),  # Haute résolution
        F.center_crop(F.resize(image, 224), 224),  # Basse résolution
        base_tf(F.adjust_brightness(image, 1.1)),  # Brightness
        base_tf(F.adjust_contrast(image, 1.1)),  # Contrast
    ]
    
    # Convertir en tensors
    tensor_list = [norm(transforms.ToTensor()(aug)) for aug in augmentations]
    batch = torch.stack(tensor_list).to(device)
    
    # Poids par modèle
    num_models = len(ensemble.models)
    if num_models == 4:
        optimal_weights = [0.30, 0.30, 0.20, 0.20]  # Avec Swin-T
    else:
        optimal_weights = [0.40, 0.35, 0.25]  # Sans Swin-T
    
    # === STRATÉGIE 1 : Moyenne pondérée ===
    total_probs = None
    all_votes = []
    
    with torch.inference_mode():
        for i, model in enumerate(ensemble.models):
            logits = model(batch)
            probs = torch.nn.functional.softmax(logits, dim=1)
            
            # Moyenne sur augmentations
            avg_probs = probs.mean(dim=0)
            
            # Collecter les votes individuels
            preds = logits.argmax(dim=1).cpu().tolist()
            all_votes.extend([preds[0]] * int(optimal_weights[i] * 10))  # Pondération voting
            
            # Accumulation pour moyenne
            weighted_probs = optimal_weights[i] * avg_probs
            if total_probs is None:
                total_probs = weighted_probs
            else:
                total_probs += weighted_probs
    
    # === DÉCISION : Moyenne ou Voting ===
    max_confidence = total_probs.max().item()
    pred_avg = int(total_probs.argmax().item())
    
    if max_confidence > 0.7:
        # Haute confiance : utiliser moyenne
        return pred_avg
    else:
        # Basse confiance : utiliser voting pour robustesse
        vote_counts = Counter(all_votes)
        pred_vote = vote_counts.most_common(1)[0][0]
        
        # Si vote et moyenne concordent : OK
        # Sinon : privilégier le vote (plus robuste)
        return pred_vote
