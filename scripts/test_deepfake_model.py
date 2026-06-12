#!/usr/bin/env python3
"""
Script de test local pour le détecteur de deepfake ONNX.

Permet de :
  1. Télécharger / charger le modèle "deepfake-vit-v2" (ONNX quantizé)
  2. Lancer une inférence sur une ou plusieurs images locales
  3. Afficher les scores bruts pour les deux labels possibles

IMPORTANT: l'ordre des labels ["Realism", "Deepfake"] dans
app/ml/onnx_processor_deepfake.py est une hypothèse basée sur la convention
ViT/HuggingFace. Utilisez ce script avec une image connue (un vrai visage de
photo, et si possible une image générée par IA) pour vérifier que
`is_fake` correspond bien à la réalité. Si les résultats sont inversés,
inversez la liste "labels" dans le model_config correspondant.

Usage:
    python scripts/test_deepfake_model.py path/to/image1.jpg [path/to/image2.png ...]
"""

import sys
from pathlib import Path

# Permettre l'import des modules de l'app depuis la racine du projet
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml.model_loader import model_manager  # noqa: E402
from app.ml.onnx_processor_deepfake import create_deepfake_detector  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_deepfake_model.py <image1> [image2 ...]")
        sys.exit(1)

    image_paths = sys.argv[1:]
    for p in image_paths:
        if not Path(p).exists():
            print(f"⚠️  Fichier introuvable: {p}")
            sys.exit(1)

    model_spec = "deepfake-vit-v2"

    print(f"Téléchargement / récupération du modèle '{model_spec}'...")
    model_path = model_manager.get_model_path(model_spec)
    print(f"Modèle disponible: {model_path}\n")

    detector = create_deepfake_detector(model_spec=model_spec, model_path=model_path)
    print("Modèle chargé.\n")

    for image_path in image_paths:
        print(f"--- {image_path} ---")
        result = detector.detect(image_path)
        print(f"  is_fake    : {result['is_fake']}")
        print(f"  confidence : {result['confidence']:.4f}")
        print(f"  scores     : {result['scores']}")
        print(f"  model_used : {result['model_used']}")
        print()

    print(
        "Vérifiez que les scores ci-dessus sont cohérents avec la nature réelle "
        "des images testées (photo réelle => Realism élevé, image générée par "
        "IA => Deepfake élevé). Si l'interprétation est inversée, modifiez "
        "l'ordre de 'labels' dans app/ml/onnx_processor_deepfake.py."
    )


if __name__ == "__main__":
    main()