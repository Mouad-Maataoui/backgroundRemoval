from typing import Dict
from pydantic import BaseModel


class DeepfakeResult(BaseModel):
    """Résultat d'une analyse de détection de deepfake"""
    is_fake: bool
    confidence: float
    scores: Dict[str, float]
    model_used: str