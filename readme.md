# Whisper Transcription Server

A GPU-accelerated speech-to-text server built with OpenAI Whisper and FastAPI.
Provides a simple HTTP API for transcription and a reference Python client for live microphone input.

---

## Requirements

* Docker + Docker Compose
* NVIDIA GPU
* NVIDIA Container Toolkit

---

## Quick Start

```bash
docker compose build
docker compose up
```

API will be available at:

```
http://localhost:9001
```

API docs:

```
http://localhost:9001/docs
```

---

## Configuration

The model is configured via environment variables.

| Variable        | Description      | Default                    |
| --------------- | ---------------- | -------------------------- |
| `WHISPER_MODEL` | Whisper model ID | `openai/whisper-tiny.en` |

### Common models

| Model  | Hugging Face ID            | Parameters   |
| ------ | -------------------------- | ------------ |
| Tiny   | `openai/whisper-tiny.en`   | 37.8M        |
| Base   | `openai/whisper-base.en`   | 72.6M      |
| Small  | `openai/whisper-small.en`  | 0.2B        |
| Medium | `openai/whisper-medium.en` | 0.8B      |
| Large  | `openai/whisper-large`     | 2B    |
| Large Turbo  | `openai/whisper-large-v3-turbo`     | 0.8B    |

---

## API Usage

### Endpoint

```
POST /transcribe
```

### Request

* `multipart/form-data`
* Field name: `file`
* Audio formats: WAV, MP3, FLAC, M4A

### Response

```json
{
  "transcription": "Transcribed text"
}
```

---

## Python Microphone Client

### Install dependencies

```bash
pip install sounddevice soundfile requests numpy
```

### Example client

```python
import queue, time, io
import numpy as np
import sounddevice as sd
import soundfile as sf
import requests

SERVER_URL = "http://localhost:9001/transcribe"
SAMPLE_RATE = 16000
CHUNK_SECONDS = 5

audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

def record():
    buffer = []
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
    ):
        last = time.time()
        while True:
            buffer.append(audio_queue.get())
            if time.time() - last >= CHUNK_SECONDS:
                send(np.concatenate(buffer))
                buffer.clear()
                last = time.time()

def send(audio):
    wav = io.BytesIO()
    sf.write(wav, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    wav.seek(0)
    r = requests.post(SERVER_URL, files={"file": ("audio.wav", wav)})
    if r.ok:
        print(">>", r.json()["transcription"])

record()
```

---

## Notes

* First startup is slow due to model download
* Each chunk is transcribed independently

---

