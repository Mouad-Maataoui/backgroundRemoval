import os
import io
import logging

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.auth_service import get_current_user
from app.models.user import User
from app.core.config import settings
from app.services.image_processing_service import ImageProcessingService
from app.services.payment_service import PaymentService
from app.models.schemas.deepfake import DeepfakeResult

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/analyze", response_model=DeepfakeResult)
async def analyze_image(
    file: UploadFile = File(..., description="Image à analyser pour détection de deepfake"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Analyse une image pour détecter si elle est un deepfake / générée par IA.
    Traitement 100% local et synchrone (résultat immédiat, pas de file Celery).
    """
    # Vérifier le nom de fichier
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nom de fichier requis"
        )

    ext = os.path.splitext(file.filename)[1].lower().strip(".")
    supported_formats = ["jpg", "jpeg", "png"]
    if ext not in supported_formats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Format non supporté. Formats acceptés: {', '.join(supported_formats)}"
        )

    # Vérifier la taille du fichier
    content = await file.read()
    file_size = len(content)
    max_size = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Fichier trop volumineux. Taille maximale: {settings.MAX_IMAGE_SIZE_MB}MB"
        )

    # Vérifier que le fichier n'est pas corrompu
    try:
        with Image.open(io.BytesIO(content)) as img:
            img.verify()
            if img.format.lower() not in ['jpeg', 'jpg', 'png']:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="L'en-tête du fichier ne correspond pas à son extension."
                )
    except (IOError, UnidentifiedImageError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le fichier image semble corrompu ou illisible."
        )

    # Vérifier et déduire les points
    payment_service = PaymentService(db)
    points_needed = settings.POINTS_COST_PER_DEEPFAKE_CHECK
    if current_user.points_balance < points_needed:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Points insuffisants. Requis: {points_needed}, disponibles: {current_user.points_balance}. Achetez plus de points."
        )

    processing_service = ImageProcessingService(
        models_name=settings.DEFAULT_AI_MODEL_DEEPFAKE,
        db_session=db,
        models_type="deepfake"
    )

    temp_path = None
    try:
        # Sauvegarder temporairement l'image pour l'inférence
        temp_path = processing_service.save_image(
            content,
            file.filename,
            subdir="uploads/deepfake_tmp"
        )

        # Déduire les points avant le traitement
        success, payment_info = payment_service.deduct_points_for_deepfake_check(current_user.id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail="Erreur lors de la déduction des points"
            )

        # Analyse de l'image
        result = await processing_service.detect_deepfake(temp_path)

        return DeepfakeResult(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse deepfake: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de l'analyse de l'image: {str(e)}"
        )
    finally:
        # Nettoyage du fichier temporaire
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning(f"Impossible de supprimer le fichier temporaire: {temp_path}")