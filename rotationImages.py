import os
from PIL import Image, ExifTags
import matplotlib.pyplot as plt

def redresser_images_et_afficher_grille(dossier):
    """
    Parcourt les images d’un dossier :
    - Redresse uniquement celles prises en portrait mais affichées couchées.
    - Affiche TOUTES les images dans une seule grille matplotlib.
    - Affiche les métadonnées EXIF uniquement pour les images redressées.
    """

    # Trouver la clé EXIF pour l'orientation
    orientation_tag = None
    for k, v in ExifTags.TAGS.items():
        if v == "Orientation":
            orientation_tag = k
            break

    fichiers = [
        f for f in os.listdir(dossier)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    images_finales = []  # liste [(Image, titre)]
    print(f"🔍 {len(fichiers)} images trouvées dans le dossier '{dossier}'\n")

    for fichier in fichiers:
        chemin = os.path.join(dossier, fichier)
        image = Image.open(chemin)

        # Lire les métadonnées EXIF
        exif_data = {}
        try:
            raw_exif = image._getexif()
            if raw_exif:
                exif_data = {ExifTags.TAGS.get(k, k): v for k, v in raw_exif.items()}
        except Exception:
            pass

        orientation = exif_data.get("Orientation", 1)
        image_corrigee = image.copy()
        modifiee = False

        # Déterminer si la photo a été prise en portrait
        prise_verticale = orientation in [6, 8]
        prise_horizontale = orientation in [1, 2, 3, 4]

        # Si elle a été prise verticale mais est couchée
        if prise_verticale and image.width > image.height:
            if orientation == 6:
                image_corrigee = image.rotate(270, expand=True)
            elif orientation == 8:
                image_corrigee = image.rotate(90, expand=True)
            else:
                image_corrigee = image.rotate(90, expand=True)
            modifiee = True

        if modifiee:
            print(f"\n✅ Image redressée : {fichier}")
            print(f"➡️ Orientation EXIF : {orientation} (prise verticale)")
            print(f"➡️ Taille avant : {image.width}x{image.height}")
            print(f"➡️ Taille après : {image_corrigee.width}x{image_corrigee.height}")

            if exif_data:
                print("\n📋 Métadonnées EXIF principales :")
                for k, v in list(exif_data.items())[:10]:
                    print(f"  - {k}: {v}")
            else:
                print("  (aucune métadonnée EXIF trouvée)")

        titre = f"{fichier}\n({'corrigée' if modifiee else 'non modifiée'})"
        images_finales.append((image_corrigee, titre))

    # --------------------------------------------
    # 🖼️ Affichage de toutes les images en grille
    # --------------------------------------------
    nb_images = len(images_finales)
    if nb_images == 0:
        print("⚠️ Aucune image trouvée dans le dossier.")
        return

    cols = 4  # nombre de colonnes dans la grille
    rows = (nb_images + cols - 1) // cols  # nombre de lignes nécessaires

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for ax, (img, titre) in zip(axes, images_finales):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(titre, fontsize=9)

    # Masquer les cases vides
    for ax in axes[len(images_finales):]:
        ax.axis("off")

    plt.suptitle("🖼️ Toutes les images après traitement", fontsize=14)
    plt.tight_layout()
    plt.show()


# Exemple d’utilisation
redresser_images_et_afficher_grille("./10Faces")  # Remplace "images" par ton dossier
