import os
import shutil

# Dossier d'origine
base_dir = "./SortedFaces"

# Dossiers de sortie
test_dir = os.path.join(base_dir, "test")
val_dir = os.path.join(base_dir, "validation")

os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for filename in os.listdir(base_dir):
    if "-" in filename and filename.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff")):
        prefix, num = filename.split("-")
        num = num.split(".")[0]  # enlève l'extension
        num = int(num)

        # Choisit le bon dossier de destination
        if 1 <= num <= 6:
            dest_root = test_dir
        elif num == 7:
            dest_root = val_dir
        else:
            continue  # ignore les autres numéros

        # Crée le dossier individuel (par numéro XXX)
        dest_folder = os.path.join(dest_root, prefix)
        os.makedirs(dest_folder, exist_ok=True)

        # Déplace le fichier
        src_path = os.path.join(base_dir, filename)
        dst_path = os.path.join(dest_folder, filename)
        shutil.move(src_path, dst_path)

print("✅ Organisation terminée : images réparties entre 'test' et 'validation'.")
