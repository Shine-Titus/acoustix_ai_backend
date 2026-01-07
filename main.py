from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import shutil
from detect import infer_audio

app = FastAPI(title="Engine Sound Anomaly Detector")

origins = [
    "http://localhost",
    "http://localhost:5500",
    "http://127.0.0.1:5500",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
ALLOWED_EXTENSIONS = {"wav", "mp3"}

os.makedirs(UPLOAD_DIR, exist_ok=True)


def validate_audio(filename: str):
    ext = filename.split(".")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only WAV or MP3 files are allowed"
        )
    return ext


@app.post("/analyse")
async def analyze_audio(file: UploadFile = File(...)):
    # Validate
    ext = validate_audio(file.filename)

    # Save uploaded file
    filename = f"{uuid.uuid4()}.{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception:
        raise HTTPException(status_code=500, detail="File save failed")

    # Run inference
    result = infer_audio(file_path)

    return {
        "result": result["status"],
        # "features": result["features"],
        "confidence": result["confidence"]
    }

@app.get("/health")
def health():
    return {"status": "ok"}
