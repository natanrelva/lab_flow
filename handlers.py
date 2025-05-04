import uuid
import time
import numpy as np
import speech_recognition as sr
import pyttsx3
import io
import pickle
from pydub import AudioSegment
import scipy.io.wavfile as wavfile
import tempfile

class DecodeHandler:
    def __init__(self):
        self.handler_id = str(uuid.uuid4())

    def process(self, chunk_id: str, pipe: dict) -> tuple:
        """Converte áudio int16 para float32."""
        for tag, data in pipe[chunk_id]:
            if tag == "input" and isinstance(data, np.ndarray):
                print(f"[DecodeHandler] chunk {chunk_id} → decoded (handler_id={self.handler_id})")
                arr = data
                if arr.ndim == 2:
                    arr = arr.mean(axis=1)
                out = arr.astype(np.float32) / 32767.0
                return {"handler_id": self.handler_id, "action": "decode", "timestamp": time.time()}, out
        return None

class SerializeHandler:
    def __init__(self):
        self.handler_id = str(uuid.uuid4())

    def process(self, chunk_id: str, pipe: dict) -> tuple:
        """Serializa o áudio float32 para bytes."""
        for tag, data in pipe[chunk_id]:
            if isinstance(tag, dict) and tag.get("action") == "decode":
                print(f"[SerializeHandler] chunk {chunk_id} → serialized audio (handler_id={self.handler_id})")
                out = pickle.dumps(data)
                return {"handler_id": self.handler_id, "action": "serialize", "timestamp": time.time()}, out
        return None

class TranscribeHandler:
    def __init__(self):
        self.handler_id = str(uuid.uuid4())
        self.recognizer = sr.Recognizer()

    def process(self, chunk_id: str, pipe: dict) -> tuple:
        """Transcreve áudio serializado para texto."""
        for tag, data in pipe[chunk_id]:
            if isinstance(tag, dict) and tag.get("action") == "serialize":
                print(f"[TranscribeHandler] chunk {chunk_id} → transcribing (handler_id={self.handler_id})")
                arr = pickle.loads(data)
                print(f"  → audio array shape: {arr.shape}, dtype: {arr.dtype}")
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                    print(f"  → writing temp WAV to: {f.name}")
                    wavfile.write(f.name, 44100, (arr * 32767).astype(np.int16))
                    with sr.AudioFile(f.name) as src:
                        audio = self.recognizer.record(src)
                        try:
                            text = self.recognizer.recognize_google(audio, language="pt-BR")
                        except Exception as e:
                            print(f"  → recognition error: {e}")
                            text = ""
                if not text:
                    print(f"  → No text recognized.")
                print(f"[TranscribeHandler] chunk {chunk_id} → got text: '{text}'")
                return {"handler_id": self.handler_id, "action": "transcribe", "timestamp": time.time()}, text
        return None

class TranslateHandler:
    def __init__(self):
        self.handler_id = str(uuid.uuid4())
        self.translation_dict = {
            "olá": "hello",
            "como você está": "how are you",
            "bom dia": "good morning",
        }

    def process(self, chunk_id: str, pipe: dict) -> tuple:
        """Traduz texto para inglês."""
        for tag, data in pipe[chunk_id]:
            if isinstance(tag, dict) and tag.get("action") == "transcribe":
                print(f"[TranslateHandler] chunk {chunk_id} → translating '{data}' (handler_id={self.handler_id})")
                out = self.translation_dict.get(data.lower(), data)
                return {"handler_id": self.handler_id, "action": "translate", "timestamp": time.time()}, out
        return None

class SerializeTextHandler:
    def __init__(self):
        self.handler_id = str(uuid.uuid4())

    def process(self, chunk_id: str, pipe: dict) -> tuple:
        """Serializa texto para bytes."""
        for tag, data in pipe[chunk_id]:
            if isinstance(tag, dict) and tag.get("action") == "translate":
                print(f"[SerializeTextHandler] chunk {chunk_id} → serialized text (handler_id={self.handler_id})")
                out = pickle.dumps(data)
                return {"handler_id": self.handler_id, "action": "serialize_text", "timestamp": time.time()}, out
        return None

class EncodeHandler:
    def __init__(self):
        self.handler_id = str(uuid.uuid4())
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 150)

    def process(self, chunk_id: str, pipe: dict) -> tuple:
        """Converte texto serializado em áudio sintetizado (int16 estéreo)."""
        for tag, data in pipe[chunk_id]:
            if isinstance(tag, dict) and tag.get("action") == "serialize_text":
                text = pickle.loads(data)
                print(f"[EncodeHandler] chunk {chunk_id} → encoding text '{text}' (handler_id={self.handler_id})")
                with io.BytesIO() as buf:
                    self.engine.save_to_file(text, buf)
                    self.engine.runAndWait()
                    buf.seek(0)
                    seg = AudioSegment.from_file(buf, format="wav")
                    arr = np.array(seg.get_array_of_samples(), dtype=np.int16)
                if len(arr) < 44100:
                    arr = np.pad(arr, (0, 44100 - len(arr)))
                else:
                    arr = arr[:44100]
                out = np.column_stack((arr, arr)).flatten()
                return {"handler_id": self.handler_id, "action": "encode", "timestamp": time.time()}, out
        return None
