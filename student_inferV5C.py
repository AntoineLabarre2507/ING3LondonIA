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

    if not models_list: raise RuntimeError("❌ Aucun modèle chargé !")
    return EnsembleModel(models_list, device), final_classes

def predict(ensemble, image, preprocess, torch):
    """
    Prédiction ultra-optimisée avec :
    - TTA étendu (10+ augmentations)
    - Multi-crop (5 crops + centre)
    - Multi-échelle (3 résolutions)
    - Temperature scaling
    - Poids optimaux
    """
    if image.mode != "RGB": image = image.convert("RGB")
    device = ensemble.device
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    # === MULTI-ÉCHELLE : 3 résolutions ===
    scales = [224, 256, 288]
    
    # === POIDS OPTIMAUX ===
    optimal_weights = [0.40, 0.35, 0.25]  # ResNet, EfficientNet, DenseNet (ajustés)
    
    # === TEMPERATURE SCALING (calibration) ===
    # Améliore la confiance des prédictions
    temperatures = [1.5, 1.0, 1.2]  # Par modèle (à affiner selon validation)
    
    all_crops = []
    
    # Pour chaque échelle
    for scale in scales:
        # Redimensionner
        img_resized = F.resize(image, scale)
        
        # === FIVE CROP + CENTER ===
        # TL, TR, BL, BR, Center
        crop_size = 224
        if scale >= crop_size:
            # Top-left
            all_crops.append(F.crop(img_resized, 0, 0, crop_size, crop_size))
            # Top-right
            all_crops.append(F.crop(img_resized, 0, scale - crop_size, crop_size, crop_size))
            # Bottom-left
            all_crops.append(F.crop(img_resized, scale - crop_size, 0, crop_size, crop_size))
            # Bottom-right
            all_crops.append(F.crop(img_resized, scale - crop_size, scale - crop_size, crop_size, crop_size))
            # Center
            all_crops.append(F.center_crop(img_resized, crop_size))
        else:
            all_crops.append(F.resize(img_resized, crop_size))
    
    # === AUGMENTATIONS HORIZONTALES ===
    augmented_crops = []
    for crop in all_crops:
        augmented_crops.append(crop)  # Original
        augmented_crops.append(F.hflip(crop))  # Flip horizontal
    
    # === AUGMENTATIONS SUPPLÉMENTAIRES sur centre uniquement ===
    center_crop = all_crops[4] if len(all_crops) > 4 else all_crops[0]
    
    # Ajustements couleur légers
    brightness_factors = [0.9, 1.0, 1.1]
    for factor in brightness_factors:
        augmented_crops.append(F.adjust_brightness(center_crop, factor))
    
    # Contraste
    contrast_factors = [0.9, 1.0, 1.1]
    for factor in contrast_factors:
        augmented_crops.append(F.adjust_contrast(center_crop, factor))
    
    # Préparer tous les tensors
    tensor_list = []
    for crop in augmented_crops:
        tensor = transforms.ToTensor()(crop)
        tensor = norm(tensor)
        tensor_list.append(tensor)
    
    # Stack en batch
    batch = torch.stack(tensor_list).to(device)
    
    total_probs = None
    
    with torch.inference_mode():
        for i, model in enumerate(ensemble.models):
            # Prédiction sur tout le batch
            logits = model(batch)
            
            # === TEMPERATURE SCALING ===
            logits = logits / temperatures[i]
            
            # Softmax et moyenne sur toutes les augmentations
            probs = torch.nn.functional.softmax(logits, dim=1).mean(dim=0)
            
            # Pondération
            weighted_probs = optimal_weights[i] * probs
            
            if total_probs is None:
                total_probs = weighted_probs
            else:
                total_probs += weighted_probs
    
    return int(total_probs.argmax().item())
