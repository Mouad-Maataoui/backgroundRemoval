Voici le rôle de chaque fichier impliqué dans la fonctionnalité de détection de deepfake :

### `app/models/schemas/deepfake.py`
Définit le schéma Pydantic `DeepfakeResult` — la structure de la réponse JSON renvoyée par l'API (`is_fake`, `confidence`, `scores`, `model_used`). Sert à valider/sérialiser automatiquement la réponse FastAPI, exactement comme les schémas existants pour `bg_remove`/`upscale`.

### `app/ml/onnx_processor_deepfake.py`
Le **cœur de l'IA**. Classe `ONNXDeepfakeDetector` (héritée de `BaseONNXProcessor`, comme `ONNXBackgroundRemover`) :
- `model_configs` : paramètres du modèle (taille d'entrée 224×224, normalisation ViT, ordre des labels `["Realism", "Deepfake"]`).
- `preprocess_image()` : convertit l'image (BGR→RGB, resize 224×224, normalisation, transposition en tenseur `(1,3,224,224)`).
- `_softmax()` : transforme les logits bruts du modèle en probabilités.
- `detect(image_path)` : charge l'image, exécute la session ONNX, calcule les scores et détermine `is_fake`.
- `create_deepfake_detector()` : factory function utilisée par le service.

### `app/api/routes/deepfake.py`
Le **endpoint API** `POST /deepfake/analyze`. Gère tout le cycle d'une requête :
- Validation du fichier (extension, taille, intégrité de l'image).
- Vérification du solde de points de l'utilisateur.
- Sauvegarde temporaire de l'image sur disque.
- Déduction des points via `PaymentService`.
- Appel à `ImageProcessingService.detect_deepfake()` pour l'inférence.
- Retour du résultat (`DeepfakeResult`) et nettoyage du fichier temporaire.

### `scripts/test_deepfake_model.py`
Script de **test en ligne de commande**, indépendant de l'API/serveur. Charge directement le détecteur ONNX, lance l'inférence sur une ou plusieurs images locales et affiche les scores bruts. Sert à vérifier que le modèle se télécharge correctement et que l'ordre des labels (`Realism`/`Deepfake`) est correct avant de faire confiance au résultat dans l'API.

## Fichiers modifiés

### `app/core/config.py`
Centralise la configuration globale de l'app (via Pydantic `Settings`). Ajouts :
- `"deepfake"` dans `MODEL_TYPES` : déclare ce nouveau type de traitement comme valide dans tout le système.
- `DEFAULT_AI_MODEL_DEEPFAKE` : nom du modèle utilisé par défaut (`deepfake-vit-v2`).
- `POINTS_COST_PER_DEEPFAKE_CHECK` : coût en points d'une analyse (1 point).

### `app/ml/model_loader.py`
Le `ModelManager`, registre central de tous les modèles ONNX téléchargeables (URL, taille, fichier local). Ajout d'un `case "deepfake":` qui enregistre deux variantes du modèle :
- `deepfake-vit-v2` (quantizé, 86MB, par défaut)
- `deepfake-vit-v2-fp32` (343MB, plus précis)

C'est ce gestionnaire qui télécharge automatiquement le `.onnx` depuis HuggingFace au premier usage et le met en cache localement.

### `app/services/image_processing_service.py`
Le **service métier** qui fait le lien entre l'API et les processeurs ONNX. Deux ajouts :
- Dans `_init_ai_processor()` : un `case "deepfake":` qui instancie `ONNXDeepfakeDetector` via `create_deepfake_detector()`, comme c'est fait pour `bg_remove`/`upscale`.
- Nouvelle méthode `detect_deepfake()` : récupère le processeur (lazy-loading), exécute `processor.detect()` dans un thread séparé (`run_in_executor`) pour ne pas bloquer la boucle asyncio, et retourne le résultat.

### `app/services/payment_service.py`
Gère les transactions de points/crédits. Ajout de `deduct_points_for_deepfake_check()` : vérifie le solde, crée une `Transaction` de type `usage` (sans `image_task_id`, car pas de tâche image associée), met à jour le solde de l'utilisateur en base, et enregistre une métrique.

### `app/api/routes/__init__.py`
Le routeur principal de l'API. Ajout de l'import du module `deepfake` et de `api_router.include_router(deepfake.router, prefix="/deepfake", ...)` — c'est ce qui rend l'endpoint `/api/v1/deepfake/analyze` accessible.

---

**Flux global d'une requête** : `deepfake.py` (route) → validation/points → `image_processing_service.py` (`detect_deepfake`) → `onnx_processor_deepfake.py` (inférence ONNX) → modèle téléchargé/géré par `model_loader.py`, paramétrage venant de `config.py`, points décomptés via `payment_service.py`, résultat formaté par `deepfake.py` (schema).

