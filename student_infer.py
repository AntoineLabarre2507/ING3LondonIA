# student_infer.py - ENSEMBLE ULTIME (ResNet + EfficientNet + TTA)
import torch
import torch.nn as nn
from torchvision import models, transforms

class EnsembleModel:
    def __init__(self, models_list, device):
        self.models = models_list
        self.device = device

    def eval(self):
        for m in self.models:
            m.eval()

    def to(self, device):
        for m in self.models:
            m.to(device)
        self.device = device
        return self

def build_model(torch, nn, models, classes=None):
    """
    Charge l'ensemble ResNet-101 + EfficientNet-B3.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_list = []
    final_classes = None

    # --- 1. Charger ResNet-101 ---
    try:
        ckpt_res = torch.load("resnet101_best_v3.pt", map_location="cpu")
        final_classes = [str(c) for c in ckpt_res["classes"]]
        
        m_res = models.resnet101(weights=None)
        m_res.fc = nn.Linear(m_res.fc.in_features, len(final_classes))
        m_res.load_state_dict(ckpt_res["model"], strict=True)
        m_res.to(device).eval()
        models_list.append(m_res)
        print("✅ ResNet-101 chargé")
    except Exception as e:
        print(f"⚠️ Erreur chargement ResNet: {e}")

    # --- 2. Charger EfficientNet-B3 ---
    try:
        ckpt_eff = torch.load("efficientnet_b3_best.pt", map_location="cpu")
        # Vérif classes identiques (important !)
        classes_eff = [str(c) for c in ckpt_eff["classes"]]
        if final_classes and classes_eff != final_classes:
             print("⚠️ ATTENTION: Classes différentes entre les modèles ! Risque d'erreur.")

        m_eff = models.efficientnet_b3(weights=None)
        m_eff.classifier[1] = nn.Linear(m_eff.classifier[1].in_features, len(classes_eff))
        m_eff.load_state_dict(ckpt_eff["model"], strict=True)
        m_eff.to(device).eval()
        models_list.append(m_eff)
        print("✅ EfficientNet-B3 chargé")
        if not final_classes: final_classes = classes_eff
    except Exception as e:
        print(f"⚠️ Erreur chargement EfficientNet: {e}")

    if not models_list:
        raise RuntimeError("❌ Aucun modèle n'a pu être chargé !")

    return EnsembleModel(models_list, device), final_classes

def predict(ensemble, image, preprocess, torch):
    """
    Prédiction d'ensemble avec TTA (Moyenne des probabilités).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    device = ensemble.device
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    # TTA : Image normale + Image retournée
    t1 = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm])
    t2 = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), norm])
    
    batch = torch.stack([t1(image), t2(image)]).to(device)

    total_probs = None
    with torch.inference_mode():
        for model in ensemble.models:
            logits = model(batch)
            probs = torch.nn.functional.softmax(logits, dim=1)
            # Moyenne des 2 vues (TTA) pour ce modèle
            avg_probs_model = probs.mean(dim=0)
            
            if total_probs is None:
                total_probs = avg_probs_model
            else:
                total_probs += avg_probs_model

    # Moyenne finale entre tous les modèles
    final_probs = total_probs / len(ensemble.models)
    return int(final_probs.argmax().item())