from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import os
import uuid
from pathlib import Path
from dotenv import load_dotenv
from models import BioSyncModel
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="BioSync API",
    description="Multimodal Bird Sensory Converter - iBC53 Indian Birds",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("FRONTEND_URL", "http://localhost:5173"),
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("🚀 Initializing BioSync model...")
try:
    model = BioSyncModel()
    logger.info("✅ BioSync model loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    model = None

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ─── ROOT ──────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "app": "BioSync API",
        "version": "1.0.0",
        "dataset": "iBC53 (25 Indian Bird Species)",
        "model": "VGG16 (Image) + CNN-LSTM (Audio)",
        "endpoints": {
            "health":         "/health",
            "species_list":   "/species-list",
            "image_to_audio": "POST /api/image-to-audio",
            "audio_to_image": "POST /api/audio-to-image",
            "download":       "GET /download/{filename}"
        }
    }


# ─── HEALTH ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status":        "healthy" if model and model.is_loaded else "degraded",
        "models_loaded": model.is_loaded if model else False,
        "model_loaded":  model.is_loaded if model else False,
        "species_count": len(model.species_list) if model else 0,
        "device":        model.device if model else "unknown",
        "version":       "1.0.0"
    }


# ─── SPECIES LIST ──────────────────────────────────────────────────────────────

@app.get("/species-list")
async def get_species_list():
    if not model:
        raise HTTPException(500, "Model not loaded")
    return {
        "total":   len(model.species_list),
        "species": [
            {
                "code":            s,
                "common_name":     model.species_metadata.get(s, {}).get("common_name", s.replace("_", " ")),
                "scientific_name": model.species_metadata.get(s, {}).get("scientific", ""),
                "family":          model.species_metadata.get(s, {}).get("family", ""),
            }
            for s in model.species_list
        ],
        "dataset": "iBC53",
        "region":  "India"
    }


# ─── SHARED LOGIC ──────────────────────────────────────────────────────────────

async def _image_to_audio(file: UploadFile):
    if not model:
        raise HTTPException(500, "Model not initialized")

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (JPG, PNG, WEBP)")

    allowed_ext = [".jpg", ".jpeg", ".png", ".webp"]
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_ext:
        raise HTTPException(400, f"Image must be one of: {', '.join(allowed_ext)}")

    image_path = None
    try:
        unique_name = f"img_{uuid.uuid4().hex}{file_ext}"
        image_path  = UPLOAD_DIR / unique_name

        content = await file.read()
        image_path.write_bytes(content)
        logger.info(f"📥 Saved upload: {image_path} ({len(content)} bytes)")

        result = model.image_to_audio(str(image_path))

        if image_path.exists():
            image_path.unlink()

        logger.info(f"✅ Predicted: {result['species']} ({result['confidence']:.2%})")

        return {
            "success":           True,
            "species_common":    result.get("common_name", result["species"]),
            "species_scientific": result.get("scientific_name", ""),
            "audio_url":         f"/download/{result['audio_file']}",
            "audio_type":        "real" if result.get("audio_type") == "real" else "synthetic",
            "confidence":        result["confidence"],
            "species":           result["species"],
            "common_name":       result.get("common_name", result["species"]),
            "scientific_name":   result.get("scientific_name", ""),
            "family":            result.get("family", "Aves"),
            "region":            result.get("region", "India (iBC53 dataset)"),
            "top3":              result.get("top3", []),
            "message":           f"Detected {result['species']} using VGG16 trained on iBC53 dataset"
        }

    except Exception as e:
        logger.error(f"❌ Image processing error: {str(e)}")
        if image_path and image_path.exists():
            image_path.unlink()
        raise HTTPException(500, f"Processing failed: {str(e)}")


async def _audio_to_image(file: UploadFile):
    if not model:
        raise HTTPException(500, "Model not initialized")

    if not file.content_type.startswith("audio/"):
        raise HTTPException(400, "File must be audio (MP3, WAV, OGG)")

    allowed_ext = [".mp3", ".wav", ".ogg", ".m4a", ".flac"]
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_ext:
        raise HTTPException(400, f"Audio must be one of: {', '.join(allowed_ext)}")

    audio_path = None
    try:
        unique_name = f"aud_{uuid.uuid4().hex}{file_ext}"
        audio_path  = UPLOAD_DIR / unique_name

        content = await file.read()
        audio_path.write_bytes(content)
        logger.info(f"📥 Saved upload: {audio_path} ({len(content)} bytes)")

        result = model.audio_to_image(str(audio_path))

        if audio_path.exists():
            audio_path.unlink()

        logger.info(f"✅ Predicted: {result['species']} ({result['confidence']:.2%})")

        return {
            "success":             True,
            "species_common":      result.get("common_name", result["species"]),
            "species_scientific":  result.get("scientific_name", ""),
            "image_url":           f"/download/{result['image_file']}",
            "image_type":          "real" if model.image_gen is not None else "placeholder",
            "confidence":          result["confidence"],
            "species":             result["species"],
            "common_name":         result.get("common_name", result["species"]),
            "scientific_name":     result.get("scientific_name", ""),
            "family":              result.get("family", "Aves"),
            "region":              result.get("region", "India (iBC53 dataset)"),
            "generated_image_url": f"/download/{result['image_file']}",
            "message":             f"Detected {result['species']} using CNN-LSTM audio classifier"
        }

    except Exception as e:
        logger.error(f"❌ Audio processing error: {str(e)}")
        if audio_path and audio_path.exists():
            audio_path.unlink()
        raise HTTPException(500, f"Processing failed: {str(e)}")


# ─── ROUTES (new + legacy) ─────────────────────────────────────────────────────

@app.post("/api/image-to-audio")
async def api_image_to_audio(file: UploadFile = File(...)):
    return await _image_to_audio(file)

@app.post("/upload-image")
async def upload_image_legacy(file: UploadFile = File(...)):
    return await _image_to_audio(file)

@app.post("/api/audio-to-image")
async def api_audio_to_image(file: UploadFile = File(...)):
    return await _audio_to_image(file)

@app.post("/upload-audio")
async def upload_audio_legacy(file: UploadFile = File(...)):
    return await _audio_to_image(file)


# ─── FILE DOWNLOAD ─────────────────────────────────────────────────────────────

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(404, "File not found. It may have expired.")

    media_types = {
        ".wav":  "audio/wav",
        ".mp3":  "audio/mpeg",
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg"
    }
    media_type = media_types.get(file_path.suffix.lower(), "application/octet-stream")
    return FileResponse(file_path, media_type=media_type, filename=filename)


# ─── ERROR HANDLERS ────────────────────────────────────────────────────────────

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(status_code=404, content={
        "error": "Endpoint not found",
        "available_endpoints": [
            "/", "/health", "/species-list",
            "/api/image-to-audio", "/api/audio-to-image",
            "/upload-image", "/upload-audio",
            "/download/{filename}"
        ]
    })

@app.exception_handler(500)
async def server_error_handler(request, exc):
    return JSONResponse(status_code=500, content={
        "error": "Internal server error",
        "message": str(exc)
    })


# ─── STARTUP ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("🚀 BioSync API Starting...")
    logger.info(f"📊 Dataset: iBC53 Indian Birds")
    logger.info(f"🐦 Species: {len(model.species_list) if model else 0}")
    logger.info(f"🖥️  Device: {model.device if model else 'unknown'}")
    logger.info("=" * 60)


# ─── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host=host, port=port, reload=True, log_level="info")