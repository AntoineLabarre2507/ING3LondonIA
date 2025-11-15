import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
import json
from datetime import datetime
from torchvision import models, transforms
from torchvision.transforms import functional as F

# Import depuis student_inferV5 sans le modifier
from student_inferV5 import EnsembleModel

class WeightedEnsembleModel(EnsembleModel):
    """Version étendue de EnsembleModel qui supporte les poids personnalisés"""
    def __init__(self, models_list, device):
        super().__init__(models_list, device)

def build_model_for_optimization():
    """Charge uniquement les 3 modèles pour l'optimisation"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_list = []
    final_classes = None
    model_names = []

    # 1. ResNet-101 V3
    try:
        ckpt = torch.load("resnet101_best_v3.pt", map_location="cpu")
        final_classes = [str(c) for c in ckpt["classes"]]
        m = models.resnet101(weights=None)
        m.fc = nn.Linear(m.fc.in_features, len(final_classes))
        m.load_state_dict(ckpt["model"], strict=True)
        m.to(device).eval()
        models_list.append(m)
        model_names.append("ResNet-101")
        print("✅ ResNet-101 chargé")
    except Exception as e: 
        print(f"⚠️ ResNet-101 non chargé: {e}")

    # 2. EfficientNet-B3
    try:
        ckpt = torch.load("efficientnet_b3_best.pt", map_location="cpu")
        m = models.efficientnet_b3(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(final_classes))
        m.load_state_dict(ckpt["model"], strict=True)
        m.to(device).eval()
        models_list.append(m)
        model_names.append("EfficientNet-B3")
        print("✅ EfficientNet-B3 chargé")
    except Exception as e: 
        print(f"⚠️ EfficientNet-B3 non chargé: {e}")

    # 3. DenseNet-121
    try:
        ckpt = torch.load("densenet121_best.pt", map_location="cpu")
        m = models.densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, len(final_classes))
        m.load_state_dict(ckpt["model"], strict=True)
        m.to(device).eval()
        models_list.append(m)
        model_names.append("DenseNet-121")
        print("✅ DenseNet-121 chargé")
    except Exception as e: 
        print(f"⚠️ DenseNet-121 non chargé: {e}")

    if not models_list: 
        raise RuntimeError("❌ Aucun modèle chargé !")
    if len(models_list) != 3:
        raise RuntimeError(f"❌ Erreur: {len(models_list)} modèles chargés, il en faut exactement 3!")
    
    return WeightedEnsembleModel(models_list, device), final_classes, model_names

def predict_with_weights(ensemble, image, weights):
    """Prédiction avec poids personnalisés pour chaque modèle"""
    if image.mode != "RGB": 
        image = image.convert("RGB")
    
    device = ensemble.device
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    base_tf = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        norm
    ])
    
    # TTA 2 vues
    batch = torch.stack([base_tf(image), base_tf(F.hflip(image))]).to(device)

    total_probs = None
    with torch.inference_mode():
        for i, model in enumerate(ensemble.models):
            logits = model(batch)
            probs = torch.nn.functional.softmax(logits, dim=1).mean(dim=0)
            weighted_probs = weights[i] * probs
            
            if total_probs is None: 
                total_probs = weighted_probs
            else: 
                total_probs += weighted_probs

    return int(total_probs.argmax().item())

class WeightOptimizer:
    def __init__(self, ensemble, val_data_path, classes, model_names):
        """
        Args:
            ensemble: Le modèle d'ensemble
            val_data_path: Chemin vers le dossier de validation (structure: val_data_path/classe/images)
            classes: Liste des noms de classes
            model_names: Liste des noms de modèles
        """
        self.ensemble = ensemble
        self.val_data_path = val_data_path
        self.classes = classes
        self.model_names = model_names
        self.validation_data = []
        self.load_validation_data()
        
    def load_validation_data(self):
        """Charge les données de validation depuis le dossier"""
        print("\n📁 Chargement des données de validation...")
        
        for class_idx, class_name in enumerate(self.classes):
            class_folder = os.path.join(self.val_data_path, class_name)
            if not os.path.exists(class_folder):
                print(f"⚠️ Dossier {class_folder} non trouvé, ignoré")
                continue
                
            images = [f for f in os.listdir(class_folder) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            
            for img_name in images:
                img_path = os.path.join(class_folder, img_name)
                self.validation_data.append((img_path, class_idx))
            
            print(f"  ✓ Classe '{class_name}': {len(images)} images")
        
        print(f"\n✅ Total: {len(self.validation_data)} images de validation chargées\n")
        
    def evaluate_weights(self, weights):
        """Évalue une combinaison de poids sur le dataset de validation"""
        correct = 0
        total = len(self.validation_data)
        
        for img_path, true_label in self.validation_data:
            try:
                image = Image.open(img_path)
                pred_label = predict_with_weights(self.ensemble, image, weights)
                if pred_label == true_label:
                    correct += 1
            except Exception as e:
                print(f"⚠️ Erreur sur {img_path}: {e}")
                total -= 1
                
        accuracy = correct / total if total > 0 else 0
        return accuracy
    
    def grid_search(self, step=0.1, top_n=10):
        """
        Recherche par grille pour 3 modèles
        Args:
            step: Pas de la grille (ex: 0.1 pour tester 0.0, 0.1, 0.2, etc.)
            top_n: Nombre de meilleures combinaisons à afficher
        """
        print(f"\n🔍 Démarrage de la recherche par grille")
        print(f"   Modèles: {', '.join(self.model_names)}")
        print(f"   Pas de recherche: {step}")
        print(f"   Images de validation: {len(self.validation_data)}")
        
        results = []
        weight_range = np.arange(0.0, 1.0 + step/2, step)
        
        # Calculer nombre total de combinaisons
        total_combinations = 0
        for w1 in weight_range:
            for w2 in weight_range:
                w3 = 1.0 - w1 - w2
                if -0.001 <= w3 <= 1.001:
                    total_combinations += 1
        
        print(f"   Combinaisons à tester: {total_combinations}\n")
        
        with tqdm(total=total_combinations, desc="🔄 Test des combinaisons") as pbar:
            for w1 in weight_range:
                for w2 in weight_range:
                    w3 = 1.0 - w1 - w2
                    
                    # Vérifier que w3 est valide
                    if w3 < -0.001 or w3 > 1.001:
                        continue
                    
                    # Normaliser les poids
                    w3 = max(0.0, min(1.0, w3))
                    weights = [w1, w2, w3]
                    total = sum(weights)
                    weights = [w/total for w in weights]
                    
                    # Évaluer cette combinaison
                    accuracy = self.evaluate_weights(weights)
                    
                    results.append({
                        'weights': [round(w, 3) for w in weights],
                        'accuracy': accuracy,
                        self.model_names[0]: weights[0],
                        self.model_names[1]: weights[1],
                        self.model_names[2]: weights[2]
                    })
                    
                    # Mettre à jour la barre de progression
                    if results:
                        best_acc = max(r['accuracy'] for r in results)
                        pbar.set_postfix({'Meilleure': f'{best_acc:.4f}'})
                    pbar.update(1)
        
        # Trier par précision décroissante
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        # Afficher les résultats
        print(f"\n{'='*80}")
        print(f"🏆 TOP {top_n} MEILLEURES COMBINAISONS DE POIDS")
        print(f"{'='*80}\n")
        
        for i, result in enumerate(results[:top_n], 1):
            print(f"#{i} - Précision: {result['accuracy']*100:.2f}%")
            for j, name in enumerate(self.model_names):
                print(f"    {name:20s}: {result['weights'][j]:.3f}")
            print()
        
        return results
    
    def random_search(self, n_iterations=1000, top_n=10):
        """
        Recherche aléatoire pour 3 modèles
        Args:
            n_iterations: Nombre de combinaisons aléatoires à tester
            top_n: Nombre de meilleures combinaisons à afficher
        """
        print(f"\n🎲 Démarrage de la recherche aléatoire")
        print(f"   Modèles: {', '.join(self.model_names)}")
        print(f"   Itérations: {n_iterations}")
        print(f"   Images de validation: {len(self.validation_data)}\n")
        
        results = []
        
        with tqdm(total=n_iterations, desc="🔄 Test aléatoire") as pbar:
            for _ in range(n_iterations):
                # Générer 3 poids aléatoires qui somment à 1
                weights_raw = np.random.dirichlet(np.ones(3))
                weights = [float(w) for w in weights_raw]
                
                # Évaluer cette combinaison
                accuracy = self.evaluate_weights(weights)
                
                results.append({
                    'weights': [round(w, 3) for w in weights],
                    'accuracy': accuracy,
                    self.model_names[0]: weights[0],
                    self.model_names[1]: weights[1],
                    self.model_names[2]: weights[2]
                })
                
                # Mettre à jour la barre de progression
                if results:
                    best_acc = max(r['accuracy'] for r in results)
                    pbar.set_postfix({'Meilleure': f'{best_acc:.4f}'})
                pbar.update(1)
        
        # Trier par précision décroissante
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        # Afficher les résultats
        print(f"\n{'='*80}")
        print(f"🏆 TOP {top_n} MEILLEURES COMBINAISONS DE POIDS (RECHERCHE ALÉATOIRE)")
        print(f"{'='*80}\n")
        
        for i, result in enumerate(results[:top_n], 1):
            print(f"#{i} - Précision: {result['accuracy']*100:.2f}%")
            for j, name in enumerate(self.model_names):
                print(f"    {name:20s}: {result['weights'][j]:.3f}")
            print()
        
        return results
    
    def save_results(self, results, filename=None):
        """Sauvegarde les résultats dans un fichier JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"weight_optimization_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Résultats sauvegardés dans: {filename}")

def main():
    """Fonction principale pour lancer l'optimisation"""
    
    # ============ CONFIGURATION ============
    # MODIFIEZ CETTE LIGNE avec le chemin vers votre dossier de validation
    VAL_DIR = "./val"
    
    # Type de recherche: "grid" ou "random"
    SEARCH_TYPE = "grid"  # ou "random"
    
    # Paramètres pour la recherche par grille
    GRID_STEP = 0.1  # Pas de 0.1 = teste 0.0, 0.1, 0.2, ..., 1.0
    
    # Paramètres pour la recherche aléatoire
    RANDOM_ITERATIONS = 1000
    
    # Nombre de meilleurs résultats à afficher
    TOP_N = 10
    
    # Sauvegarder les résultats ?
    SAVE_RESULTS = True
    # =======================================
    
    print("="*80)
    print("🚀 OPTIMISATION DES POIDS D'ENSEMBLE")
    print("="*80)
    
    try:
        # Vérifier que le dossier de validation existe
        if not os.path.exists(VAL_DIR):
            raise FileNotFoundError(
                f"❌ Le dossier de validation '{VAL_DIR}' n'existe pas!\n"
                "   Veuillez mettre à jour la variable VAL_DIR dans le script."
            )
        
        # Charger les modèles
        print("\n📦 Chargement des modèles...")
        ensemble, classes, model_names = build_model_for_optimization()
        print(f"✅ {len(model_names)} modèles chargés avec succès")
        print(f"📊 Nombre de classes: {len(classes)}")
        
        # Créer l'optimiseur
        optimizer = WeightOptimizer(ensemble, VAL_DIR, classes, model_names)
        
        # Lancer la recherche
        if SEARCH_TYPE == "grid":
            results = optimizer.grid_search(step=GRID_STEP, top_n=TOP_N)
        elif SEARCH_TYPE == "random":
            results = optimizer.random_search(n_iterations=RANDOM_ITERATIONS, top_n=TOP_N)
        else:
            raise ValueError(f"Type de recherche invalide: {SEARCH_TYPE}")
        
        # Sauvegarder les résultats
        if SAVE_RESULTS:
            optimizer.save_results(results)
        
        # Afficher le meilleur résultat
        best = results[0]
        print(f"\n{'='*80}")
        print("🎯 MEILLEUR RÉSULTAT")
        print(f"{'='*80}")
        print(f"Précision: {best['accuracy']*100:.2f}%")
        print(f"Poids optimaux:")
        for i, name in enumerate(model_names):
            print(f"  • {name:20s}: {best['weights'][i]:.3f}")
        print(f"{'='*80}\n")
        
    except FileNotFoundError as e:
        print(f"\n{e}")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
