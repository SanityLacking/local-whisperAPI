from fastapi import FastAPI, UploadFile, Form, HTTPException, Header, Depends
from typing import Optional
from faster_whisper import WhisperModel
import os
import uvicorn
app = FastAPI()

# Load Whisper models into memory for quick access
MODELS = {
    "base": WhisperModel("base",device="cpu"),
    # "small": WhisperModel("small",device="cuda"),
    # Add more models as needed
}

# API key for authentication
API_KEY = "sk-your_secret_api_key"

def verify_api_key(Authorization: str = Header(...)):
    if Authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API Key")

@app.post("/v1/audio/transcriptions", dependencies=[Depends(verify_api_key)])
async def transcribe_audio(
    file: UploadFile,
    model: str = "base",
    language: Optional[str] = "en",
    temperature: Optional[float] = 0.0
):
    """
    Emulates the OpenAI Whisper transcription endpoint.
    """
    print(model)
    # Check if the requested model is available
    if model not in MODELS:
        raise HTTPException(status_code=400, detail="Model not found")

    # Save the uploaded file temporarily
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as f:
        f.write(await file.read())
    print("file written")
    try:
        # Transcribe audio using the selected model
        whisper_model = MODELS[model]
        options = {"temperature": temperature}
        if language:
            options["language"] = language
        result, info = whisper_model.transcribe(temp_file_path, **options)
        # Format response similar to OpenAI Whisper
        response = {
            "text": list(result),
            "language": info["language"] if language is None else language,
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during transcription: {str(e)}")

    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
def list_models():
    """
    Lists available models for transcription.
    """
    return {"models": list(MODELS.keys())}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")