import uuid
import numpy as np
import wave
from handlers import DecodeHandler, SerializeHandler, TranscribeHandler

# Constantes de áudio
CHANNELS = 1
RATE = 44100
CHUNK = 4924

def load_audio_from_file(file_path):
    """Carrega áudio de um arquivo WAV e retorna os dados como um array NumPy."""
    with wave.open(file_path, 'rb') as wf:
        if wf.getnchannels() != CHANNELS or wf.getframerate() != RATE:
            raise ValueError("O arquivo de áudio não tem a configuração esperada (mono, 44100Hz).")

        # Lê os dados do áudio e converte para NumPy
        audio_bytes = wf.readframes(wf.getnframes())
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)

    return audio_data


def main():
    # 1) Carrega o arquivo de áudio local
    file_path = "sample_pt_converted.wav"  # Substitua pelo caminho do arquivo local
    arr = load_audio_from_file(file_path)
    chunk_id = str(uuid.uuid4())
    pipe = {chunk_id: [("input", arr)]}

    # 2) Decode
    dec = DecodeHandler()
    tag1, decoded = dec.process(chunk_id, pipe)
    pipe[chunk_id].append((tag1, decoded))

    # 3) Serialize
    ser = SerializeHandler()
    tag2, serialized = ser.process(chunk_id, pipe)
    pipe[chunk_id].append((tag2, serialized))

    # 4) Transcribe
    trans = TranscribeHandler()
    tag3, text = trans.process(chunk_id, pipe)
    print("Texto transcrito (do arquivo):", repr(text))


if __name__ == "__main__":
    main()
