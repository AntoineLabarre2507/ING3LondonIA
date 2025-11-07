# student_infer.py - Ensemble de modèles
def build_model(torch, nn, models, classes):
    """
    Charge DEUX modèles : EfficientNet-B3 et ResNet-50.
    Retourne un tuple (models_list, classes, torch_ref)
    """
    # Charger EfficientNet-B3
    try:
        ckpt_eff = torch.load("efficientnet_b3_best.pt", map_location="cpu")
        classes = [str(c) for c in ckpt_eff["classes"]]
        
        model_eff = models.efficientnet_b3(weights=None)
        num_ftrs = model_eff.classifier[1].in_features
        model_eff.classifier[1] = nn.Linear(num_ftrs, len(classes))
        model_eff.load_state_dict(ckpt_eff["model"], strict=True)
        model_eff.eval()
        print("✅ EfficientNet-B3 chargé")
    except Exception as e:
        print(f"⚠️  EfficientNet-B3 non trouvé: {e}")
        model_eff = None
    
    # Charger ResNet-50
    model_resnet = None
    try:
        ckpt_resnet = torch.load("resnet50_best.pt", map_location="cpu")
        model_resnet = models.resnet50(weights=None)
        num_ftrs = model_resnet.fc.in_features
        model_resnet.fc = nn.Linear(num_ftrs, len(classes))
        model_resnet.load_state_dict(ckpt_resnet["model"], strict=True)
        model_resnet.eval()
        print("✅ ResNet-50 chargé")
    except Exception as e:
        print(f"⚠️  ResNet-50 non trouvé (optionnel): {e}")
    
    # Retourner les modèles en tuple
    models_list = [m for m in [model_eff, model_resnet] if m is not None]
    
    # Créer un objet "wrapper" qui a une méthode .eval()
    class EnsembleWrapper:
        def __init__(self, models_list, classes, torch_ref):
            self.models_list = models_list
            self.classes = classes
            self.torch_ref = torch_ref
        
        def eval(self):
            for m in self.models_list:
                m.eval()
            return self
    
    wrapper = EnsembleWrapper(models_list, classes, torch)
    return wrapper, classes


def predict(model_wrapper, image, preprocess, torch):
    """
    Prédit en utilisant l'ENSEMBLE de modèles.
    Combine les probabilités et retourne la classe avec la plus haute prob moyenne.
    
    Args:
        model_wrapper: EnsembleWrapper contenant les modèles
        image: Image PIL
        preprocess: Transformations
        torch: Module torch
    
    Returns:
        int: Index de la classe prédite
    """
    models_list = model_wrapper.models_list
    
    # S'assurer que l'image est en RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Prétraiter l'image
    x = preprocess(image).unsqueeze(0)
    
    # Accumuler les probabilités de tous les modèles
    ensemble_probs = None
    
    with torch.inference_mode():
        for i, model in enumerate(models_list):
            logits = model(x)
            probs = torch.softmax(logits, dim=1)  # Probabilités [0, 1]
            
            if ensemble_probs is None:
                ensemble_probs = probs.clone()
            else:
                ensemble_probs += probs
    
    # Moyenne des probabilités
    ensemble_probs = ensemble_probs / len(models_list)
    
    # Prendre la classe avec la plus haute probabilité moyenne
    pred_idx = int(ensemble_probs.argmax(1).item())
    
    return pred_idx