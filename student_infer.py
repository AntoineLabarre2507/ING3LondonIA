# student_infer.py
def build_model(torch, nn, models, classes):
    """
    Charge le modèle SqueezeNet entraîné.
    """
    ckpt = torch.load("C:/Cours LONDON/AI/Python/projets/squeezenet_best.pt", map_location="cpu")
    classes = [str(c) for c in ckpt["classes"]]

    # Utilise SqueezeNet au lieu de GoogLeNet
    m = models.squeezenet1_0(weights=None)
    m.classifier[1] = torch.nn.Conv2d(512, len(classes), kernel_size=1)
    m.num_classes = len(classes)
    m.load_state_dict(ckpt["model"], strict=True)  # strict=True pour s'assurer que tout est chargé correctement
    m.eval()
    return m, classes

def predict(model, image, preprocess, torch):
    """
    Prédit la classe d'une image PIL.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    x = preprocess(image).unsqueeze(0)  # Ajoute une dimension batch
    with torch.inference_mode():
        logits = model(x)
        return int(logits.argmax(1).item())
