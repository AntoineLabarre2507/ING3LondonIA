import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os, time, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

def main():
    # ---- Configuration V3 (Retour aux sources optimisé) ----
    TRAIN_DIR = 'train'
    VAL_DIR   = 'val'
    BATCH_SIZE = 16
    EPOCHS = 35          # Suffisant pour ResNet
    LR = 1e-3            # On revient au LR standard de ResNet
    NUM_WORKERS = 2
    SEED = 42
    MODEL_NAME = "resnet101"
    BEST_PATH = f"{MODEL_NAME}_best_v3.pt"

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device : {device} | Modèle : {MODEL_NAME} (Optimisé)")

    # ============================================================
    # 1. Data Augmentation (Celle qui marchait bien avant + légères retouches)
    # ============================================================
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), # On revient à 0.5, ResNet aime bien
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_tf)
    num_classes = len(train_ds.classes)
    print(f"📁 Classes : {num_classes} (Doit être 114)")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # ============================================================
    # 2. Modèle : ResNet-101
    # ============================================================
    print(f"🏗️ Chargement de {MODEL_NAME}...")
    weights = models.ResNet101_Weights.IMAGENET1K_V2
    model = models.resnet101(weights=weights)

    # Remplacement de la fc
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.to(device)

    # Fine-tuning : On dégele layer4 et fc (ta meilleure config précédente)
    for name, param in model.named_parameters():
        if 'layer4' in name or 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    print("🔒 ResNet-101 : Gelé sauf layer4 et fc.")

    # ============================================================
    # 3. Optimisation
    # ============================================================
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-3)
    # ON GARDE LE LABEL SMOOTHING ! C'est l'ingrédient secret pour battre tes 77%
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3, verbose=True)

    # ============================================================
    # 4. Boucle
    # ============================================================
    best_acc = 0.0
    patience, patience_max = 0, 10

    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        tr_correct, tr_total = 0, 0
        loop = tqdm(train_loader, leave=False, desc=f"Epoch {epoch}/{EPOCHS}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            tr_correct += (outputs.argmax(1) == labels).sum().item()
            tr_total += imgs.size(0)
            loop.set_postfix(loss=loss.item())
        train_acc = tr_correct / tr_total

        # Val
        model.eval()
        val_correct, val_total = 0, 0
        with torch.inference_mode():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += imgs.size(0)
        val_acc = val_correct / val_total
        
        scheduler.step(val_acc)
        print(f"Epoch {epoch:02d} | Tr Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            torch.save({"model": model.state_dict(), "classes": train_ds.classes}, BEST_PATH)
            print(f"💾 NEW BEST: {best_acc:.4f}")
        else:
            patience += 1
            if patience >= patience_max:
                print("🛑 Early Stopping")
                break

    print(f"\n🏁 Terminé. Meilleur score V3 : {best_acc:.4f}")

if __name__ == "__main__":
    main()