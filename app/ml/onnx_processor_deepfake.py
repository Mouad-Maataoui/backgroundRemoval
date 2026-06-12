import logging
from typing import Dict

import cv2
import numpy as np

from app.ml.base_onnx_processor import BaseONNXProcessor

logger = logging.getLogger(__name__)


class ONNXDeepfakeDetector(BaseONNXProcessor):
    """
    Processeur ONNX pour la détection de deepfake / image générée par IA.

    Modèle par défaut: onnx-community/Deep-Fake-Detector-v2-Model-ONNX
    (ViT-base, fine-tuné pour classer une image en "Realism" ou "Deepfake")
    """

    def __init__(self, model_path: str = None, model_spec: str = "deepfake-vit-v2"):
        # Configuration par modèle
        self.model_configs = {
            "deepfake-vit-v2": {
                "input_size": (224, 224),
                # ViTImageProcessor par défaut: mean=std=0.5 (normalisation [-1, 1])
                "normalize_mean": [0.5, 0.5, 0.5],
                "normalize_std": [0.5, 0.5, 0.5],
                # Ordre des labels en sortie du modèle (index 0, index 1, ...)
                # ATTENTION: à vérifier avec le script de test (cf. test_ai_model /
                # scripts/test_deepfake_model.py) et ajuster si besoin.
                "labels": ["Deepfake", "Realism"],
            },
            "deepfake-vit-v2-fp32": {
                "input_size": (224, 224),
                "normalize_mean": [0.5, 0.5, 0.5],
                "normalize_std": [0.5, 0.5, 0.5],
                "labels": ["Deepfake", "Realism"],
            },
        }

        if model_spec not in self.model_configs:
            raise ValueError(f"Type de modèle non supporté: {model_spec}")

        self.config = self.model_configs[model_spec]
        self.spec = model_spec
        self.models_type = "deepfake"

        super().__init__(model_path=str(model_path), model_spec=model_spec)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Préprocessing de l'image pour l'inférence ONNX (classification ViT)

        Args:
            image: Image numpy (H, W, C) en format BGR

        Returns:
            Tensor d'entrée normalisé (1, C, H, W)
        """
        # BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize direct à la taille d'entrée du modèle (pas de padding nécessaire
        # pour de la classification globale)
        input_height, input_width = self.config["input_size"]
        image_resized = cv2.resize(
            image, (input_width, input_height), interpolation=cv2.INTER_LINEAR
        )

        # Normalisation [0, 255] -> [0, 1] -> mean/std du modèle
        image_normalized = image_resized.astype(np.float32) / 255.0
        mean = np.array(self.config["normalize_mean"], dtype=np.float32)
        std = np.array(self.config["normalize_std"], dtype=np.float32)
        image_normalized = (image_normalized - mean) / std

        # (H, W, C) -> (1, C, H, W)
        tensor = np.transpose(image_normalized, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        return tensor

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def detect(self, image_path: str) -> Dict:
        """
        Analyse une image et retourne un score de probabilité deepfake.

        Args:
            image_path: Chemin vers l'image à analyser

        Returns:
            dict avec is_fake, confidence, scores par label, et le modèle utilisé
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Impossible de charger l'image: {image_path}")

            input_tensor = self.preprocess_image(image)

            logger.info(f"Inférence ONNX deepfake ({self.spec})...")
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            logits = np.array(outputs[0]).reshape(-1)  # (num_classes,)

            probs = self._softmax(logits)
            labels = self.config["labels"]

            scores = {labels[i]: float(probs[i]) for i in range(len(labels))}
            fake_score = scores.get("Deepfake", 0.0)
            real_score = scores.get("Realism", 0.0)

            is_fake = fake_score > real_score

            result = {
                "is_fake": bool(is_fake),
                "confidence": float(max(probs)),
                "scores": scores,
                "model_used": self.spec,
            }

            logger.info(f"Résultat détection deepfake: {result}")
            return result

        except Exception as e:
            logger.error(f"Erreur détection deepfake: {str(e)}")
            raise


def create_deepfake_detector(model_spec: str = "deepfake-vit-v2", model_path: str = None) -> ONNXDeepfakeDetector:
    """
    Crée un processeur de détection de deepfake

    Args:
        model_spec: Type de modèle ('deepfake-vit-v2', 'deepfake-vit-v2-fp32')
        model_path: Chemin personnalisé vers le modèle

    Returns:
        Instance du processeur ONNX
    """
    return ONNXDeepfakeDetector(model_path=model_path, model_spec=model_spec)