import os
import shutil
import wave
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Tuple

import librosa

import av
import numpy as np
import requests
import senko
import sherpa_onnx
import soundfile as sf
from fastapi import FastAPI, File, UploadFile
from math import ceil
from pydantic import BaseModel

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


def diarize(diarizer, wav_path: str):
    print("[TASK] Speaker Diarization")
    diar_data = diarizer.diarize(wav_path, generate_colors=True)
    diar_segments = diar_data.get("merged_segments", [])
    print(f"Found {len(diar_segments)} speaker segments")
    for seg in diar_segments[:3]:
        print(f"      [{seg['start']:.2f}s - {seg['end']:.2f}s] Speaker {seg['speaker']}")

    return diar_segments


def read_wave(wav_path: str) -> Tuple[np.ndarray, int]:
    with wave.open(wav_path) as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)

        samples_float32 = samples_float32 / 32768
        return samples_float32, f.getframerate()


def resample_audio(input_path: str,
                   output_path: str,
                   target_rate: int = 16000):
    # Load + resample in one step (mono=True forces mono)
    audio, _ = librosa.load(input_path, sr=target_rate, mono=True)

    # Write as 16-bit PCM WAV
    sf.write(output_path, audio, target_rate, subtype="PCM_16")



def transcript(recognizer, wav_path):
    audio, sample_rate = read_wave(wav_path)

    print("[TASK] Running speech recognition...")
    stream = recognizer.create_stream()

    stream.accept_waveform(16000, audio)

    recognizer.decode_stream(stream)
    print(f"ASR tokens: {len(stream.result.tokens)}")

    return stream


def combine_diar_asr_res(diar_res, asr_res):
    print("[TASK] Combining diarization + ASR...")
    output = []
    for seg in diar_res:
        start, end, speaker = float(seg["start"]), float(seg["end"]), str(seg["speaker"])
        tokens = [tok for tok, ts in zip(asr_res.result.tokens, asr_res.result.timestamps)
                  if start <= float(ts) <= end]
        text = "".join(tokens).strip()
        if text:
            output.append(f"[{start:.2f}s - {end:.2f}s] {speaker}: {text}")

    transcript = "\n".join(output) if output else asr_res.result.text.strip()

    print("\n[RESULT]")
    print(transcript)

    return transcript


diarizer = None
recognizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global diarizer
    global recognizer

    # Load the ML model
    print("Setting up Diarizer")
    diarizer = senko.Diarizer(device="auto", warmup=True, quiet=False)

    print("Setting up Recognizer")

    MODEL_DIR = "models/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09"

    if MODEL_DIR == "models/sherpa-onnx-zipformer-vi-2025-04-20":
        raise Exception('Not coded')
        # recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        #     encoder=f"{MODEL_DIR}/encoder-epoch-12-avg-8.onnx",
        #     decoder=f"{MODEL_DIR}/decoder-epoch-12-avg-8.onnx",
        #     joiner=f"{MODEL_DIR}/joiner-epoch-12-avg-8.onnx",
        #     tokens=f"{MODEL_DIR}/tokens.txt",
        #     bpe_vocab=f"{MODEL_DIR}/bpe.model",
        #     num_threads=4,
        #     debug=True,
        # )
    elif MODEL_DIR == "models/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09":
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=f"{MODEL_DIR}/encoder.int8.onnx",
            decoder=f"{MODEL_DIR}/decoder.onnx",
            joiner=f"{MODEL_DIR}/joiner.int8.onnx",
            tokens=f"{MODEL_DIR}/tokens.txt",
            bpe_vocab=f"{MODEL_DIR}/bpe.model",
            num_threads=4,
            debug=True,
        )
    else:
        raise Exception('Not coded')

    yield


app = FastAPI(lifespan=lifespan)


class WavURLRequest(BaseModel):
    url: str


@app.post("/audio2text")
async def audio2text(file: UploadFile = File(...)):
    # [1] Get file upload and save file

    wav_path = f"datastore/{file.filename}"

    with open(wav_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    wav_16k_path = f"datastore/16k/{file.filename}"

    resample_audio(wav_path, wav_16k_path, target_rate=16000)

    wav_directory = wav_path.split('.')[0]

    if os.path.exists(wav_directory):
        shutil.rmtree(wav_directory)

    os.mkdir(wav_directory)

    # [2] Cut wav into chunks, 60s each

    if not os.listdir(wav_directory):
        audio, sample_rate = read_wave(wav_16k_path)
        duration = len(audio) / sample_rate

        chunk_samples = 60 * sample_rate

        for i in range(ceil(len(audio) / chunk_samples)):
            chunk = audio[i * chunk_samples: (i + 1) * chunk_samples]

            start_time = i * chunk_samples / sample_rate
            end_time = min(duration, (i + 1) * chunk_samples / sample_rate)

            chunk_path = f"{wav_directory}/{i}.wav"

            sf.write(chunk_path, chunk, sample_rate)

    output_path = wav_path.split('.')[0] + ".txt"
    if os.path.exists(output_path):
        os.remove(output_path)
    print(f"Writing to {output_path}")

    # [3] Run Diarization and ASR for each chunk. Write to output_path

    for i in range(len(os.listdir(wav_directory))):
        chunk_path = f"{wav_directory}/{i}.wav"

        diar_res = diarize(diarizer, chunk_path)
        asr_res = transcript(recognizer, chunk_path)

        res_txt = combine_diar_asr_res(diar_res, asr_res)

        with open(output_path, "a+") as f:
            f.write(res_txt + "\n")

    with open(output_path, "r") as f:
        return {"result": f.read()}


@app.get("/", response_class=HTMLResponse)
async def upload_form():
    return """
    <html>
        <style>
          html, body {
            font-family: Arial, Helvetica, sans-serif;
          }
        </style>
        <body>
            <h1>Upload Audio</h1>
            <form id="uploadForm">
                <input id="fileInput" name="file" type="file" accept=".wav,.mp3">
                <button type="submit">Process</button>
            </form>
            <div id="response" style="margin-top: 20px; padding: 15px; border: 1px solid #ccc; white-space: pre-wrap; line-height: 1.6;"></div>

            <script>
                document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                    e.preventDefault();

                    const fileInput = document.getElementById('fileInput');
                    const formData = new FormData();
                    formData.append('file', fileInput.files[0]);
                    
                    document.getElementById('response').innerText = "Processing..."

                    const response = await fetch('/audio2text', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();

                    // Display the long text
                    document.getElementById('response').innerText = data.result;
                });
            </script>
        </body>
    </html>
    """