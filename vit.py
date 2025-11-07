import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os, time, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from pathlib import Path
from collections import Counter

def main():
    # ---- Configuration ----
    train_dir = 'train'
    val_dir   = 'val'
    batch_size = 16
    epochs = 50
    lr = 1e-3
    num_workers = 2
    seed = 42
    
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # ============================================================
    # Préparation des données (identique)
    # ============================================================
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    def is_ok(path):
        name = Path(path).name
        if name.startswith("._") or name.startswith("."):
            return False
        return name.lower().endswith((".jpg", ".jpeg", ".png"))

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf, is_valid_file=is_ok)
    val_ds   = datasets.ImageFolder(val_dir,   transform=val_tf,  is_valid_file=is_ok)
    
    train_labels = [y for _, y in train_ds.samples]
    print("Répartition des classes (train) :", Counter(train_labels))
    
    num_classes = len(train_ds.classes)
    print(f"📁 Classes trouvées : {num_classes}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ============================================================
    # Modèle ResNet-50
    # ============================================================
    from torchvision.models import resnet50, ResNet50_Weights
    
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    
    # Remplacer la dernière couche
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.to(device)
    
    # Geler les premières couches, dégeler les dernières
    for name, param in model.named_parameters():
        if 'layer4' in name or 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    print(f"🔒 ResNet-50 : gélé sauf layer4 et fc")
    
    # Optimiseur et scheduler
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # ============================================================
    # Entraînement
    # ============================================================
    def run_epoch(loader, train=False):
        model.train(train)
        total, correct, loss_sum = 0, 0, 0.0
        loop = tqdm(loader, leave=False)
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.set_grad_enabled(train):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                if train:
                    loss.backward()
                    optimizer.step()
            
            preds = outputs.argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            loss_sum += loss.item() * labels.size(0)
            loop.set_description(f"{'Train' if train else 'Val'} | loss {loss.item():.4f}")
        
        return loss_sum / total, correct / total

    # ============================================================
    # Boucle principale
    # ============================================================
    best_acc = 0.0
    best_path = "resnet50_best.pt"
    patience_counter = 0
    patience_max = 10
    start = time.time()
    
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(train_loader, train=True)
        va_loss, va_acc = run_epoch(val_loader, train=False)
        scheduler.step(va_acc)
        
        overfit = tr_acc - va_acc
        print(f"Epoch {epoch:02d}/{epochs} | train {tr_acc:.4f} | val {va_acc:.4f} | overfit {overfit:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if va_acc > best_acc:
            best_acc = va_acc
            patience_counter = 0
            torch.save({
                "model": model.state_dict(),
                "classes": train_ds.classes
            }, best_path)
            print(f"💾 Nouveau meilleur modèle sauvegardé ({best_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience_max:
                print(f"\n🛑 Early stopping après {epoch} epochs (pas d'amélioration depuis {patience_max} epochs)")
                break
    
    mins = (time.time() - start) / 60
    print(f"\n✅ Terminé en {mins:.1f} min | meilleure val acc {best_acc:.4f} | sauvegardé sous {best_path}")

if __name__ == "__main__":
    main()