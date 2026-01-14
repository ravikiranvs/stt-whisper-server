import io
import os
import torch
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import WhisperProcessor, WhisperForConditionalGeneration

app = FastAPI(title="Whisper Transcription Server")

MODEL_NAME = os.getenv("WHISPER_MODEL", "openai/whisper-tiny.en")

processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.eval()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file")

    try:
        audio_bytes = await file.read()
        audio_array, sampling_rate = sf.read(
            io.BytesIO(audio_bytes),
            dtype="float32"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")

    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    inputs = processor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt"
    )

    input_features = inputs.input_features.to(DEVICE)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    return {"transcription": transcription}
