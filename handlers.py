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
                # Converte estéreo para mono
                if data.ndim == 2:
                    data = data.mean(axis=1)
                processed_data = data.astype(np.float32) / 32767.0
                tag = {
                    "handler_id": self.handler_id,
                    "action": "decode",
                    "timestamp": time.time()
                }
                return tag, processed_data
        return None


class SerializeHandler:
    def __init__(self):
        self.handler_id = str(uuid.uuid4())

    def process(self, chunk_id: str, pipe: dict) -> tuple:
        """Serializa o áudio float32 para bytes."""
        for tag, data in pipe[chunk_id]:
            if isinstance(tag, dict) and tag.get("action") == "decode":
                processed_data = pickle.dumps(data)
                tag = {
                    "handler_id": self.handler_id,
                    "action": "serialize",
                    "timestamp": time.time()
                }
                return tag, processed_data
        return None


class TranscribeHandler:
    def __init__(self):
        self.handler_id = str(uuid.uuid4())
        self.recognizer = sr.Recognizer()

    def process(self, chunk_id: str, pipe: dict) -> tuple:
        """Transcreve áudio serializado para texto."""
        for tag, data in pipe[chunk_id]:
            if isinstance(tag, dict) and tag.get("action") == "serialize":
                audio_data = pickle.loads(data)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
                    wavfile.write(temp_wav.name, 44100, (audio_data * 32767).astype(np.int16))
                    with sr.AudioFile(temp_wav.name) as source:
                        audio = self.recognizer.record(source)
                        try:
                            text = self.recognizer.recognize_google(audio, language="en-US")
                            tag = {
                                "handler_id": self.handler_id,
                                "action": "transcribe",
                                "timestamp": time.time()
                            }
                            return tag, text
                        except (sr.UnknownValueError, sr.RequestError):
                            return None
        return None


class TranslateHandler:
    def __init__(self):
        self.handler_id = str(uuid.uuid4())
        self.translation_dict = {
            "hello": "olá",
            "how are you": "como você está",
            "good morning": "bom dia",
        }

    def process(self, chunk_id: str, pipe: dict) -> tuple:
        """Traduz texto para português."""
        for tag, data in pipe[chunk_id]:
            if isinstance(tag, dict) and tag.get("action") == "transcribe":
                translated_text = self.translation_dict.get(data.lower(), data)
                tag = {
                    "handler_id": self.handler_id,
                    "action": "translate",
                    "timestamp": time.time()
                }
                return tag, translated_text
        return None


class SerializeTextHandler:
    def __init__(self):
        self.handler_id = str(uuid.uuid4())

    def process(self, chunk_id: str, pipe: dict) -> tuple:
        """Serializa texto para bytes."""
        for tag, data in pipe[chunk_id]:
            if isinstance(tag, dict) and tag.get("action") == "translate":
                processed_data = pickle.dumps(data)
                tag = {
                    "handler_id": self.handler_id,
                    "action": "serialize_text",
                    "timestamp": time.time()
                }
                return tag, processed_data
        return None


class EncodeHandler:
    def __init__(self):
        self.handler_id = str(uuid.uuid4())
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 150)

    def process(self, chunk_id: str, pipe: dict) -> tuple:
        """Converte texto serializado em áudio sintetizado (int16)."""
        for tag, data in pipe[chunk_id]:
            if isinstance(tag, dict) and tag.get("action") == "serialize_text":
                text = pickle.loads(data)
                if not text:
                    processed_data = np.zeros(44100, dtype=np.int16)
                else:
                    with io.BytesIO() as audio_io:
                        self.engine.save_to_file(text, audio_io)
                        self.engine.runAndWait()
                        audio_io.seek(0)
                        audio_segment = AudioSegment.from_file(audio_io, format="wav")
                        audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)
                        if len(audio_data) < 44100:
                            audio_data = np.pad(audio_data, (0, 44100 - len(audio_data)))
                        elif len(audio_data) > 44100:
                            audio_data = audio_data[:44100]
                        processed_data = audio_data
                # Converte mono para estéreo
                processed_data = np.column_stack((processed_data, processed_data)).flatten()
                tag = {
                    "handler_id": self.handler_id,
                    "action": "encode",
                    "timestamp": time.time()
                }
                return tag, processed_data
        return None


