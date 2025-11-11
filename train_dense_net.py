import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os, time, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

def main():
    # ---- Configuration DenseNet-121 ----
    TRAIN_DIR = 'train'
    VAL_DIR   = 'val'
    BATCH_SIZE = 16
    EPOCHS = 35
    LR = 1e-3
    NUM_WORKERS = 2
    SEED = 42
    MODEL_NAME = "densenet121"
    BEST_PATH = f"{MODEL_NAME}_best.pt"

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device : {device} | Modèle : {MODEL_NAME} (Ensemble Member 3 - The Wildcard)")

    # Data Augmentation standard qui marche bien
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(mean, std)
    ])

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Modèle DenseNet-121
    print(f"🏗️ Chargement de {MODEL_NAME}...")
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    
    # Remplacement du classifier (il s'appelle 'classifier' sur DenseNet)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(train_ds.classes))
    model.to(device)

    # Fine-tuning : Dégeler le dernier bloc dense (denseblock4) et le classifier
    for name, param in model.named_parameters():
        if 'denseblock4' in name or 'classifier' in name or 'norm5' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    print("🔒 DenseNet : Gelé sauf dernier bloc.")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3, verbose=True)

    # Boucle
    best_acc = 0.0
    patience, patience_max = 0, 10
    for epoch in range(1, EPOCHS + 1):
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

        model.eval()
        val_correct, val_total = 0, 0
        with torch.inference_mode():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                val_correct += (model(imgs).argmax(1) == labels).sum().item()
                val_total += imgs.size(0)
        val_acc = val_correct / val_total
        
        scheduler.step(val_acc)
        overfit = train_acc - val_acc
        print(f"Epoch {epoch:02d} | Tr Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Overfit: {overfit:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            torch.save({"model": model.state_dict(), "classes": train_ds.classes}, BEST_PATH)
            print(f"💾 NEW BEST: {best_acc:.4f}")
        else:
            patience += 1
            if patience >= patience_max: break

    print(f"🏁 Terminé. Meilleur DenseNet : {best_acc:.4f}")

if __name__ == "__main__":
    main()