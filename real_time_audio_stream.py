import pyaudio
import numpy as np
import wave
import uuid
import time
from collections import deque, defaultdict
from typing import Any, Dict, List

class RealTimeAudioStream:
    def __init__(self, rate=44100, chunk_size=44100, channels_in=1, channels_out=2, format=pyaudio.paInt16, input_device_index=None, output_device_index=None):
        # Inicializando parâmetros de áudio
        self.rate = rate
        self.chunk_size = chunk_size  # 1 segundo de áudio para transcrição
        self.channels_in = channels_in  # Mono para microfone
        self.channels_out = channels_out  # Estéreo para BlackHole/Multi-Output
        self.format = format
        self.p = pyaudio.PyAudio()
        self.handlers = []
        self.pipe = defaultdict(list)  # Pipe compartilhado: {chunk_id: [(tag, data), ...]}
        self.required_handlers = 0
        self.buffer = deque(maxlen=10)  # Buffer para chunks capturados

        # Abrindo fluxo de captura de áudio com callback
        self.input_stream = self.p.open(
            format=self.format,
            channels=self.channels_in,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=input_device_index,
            stream_callback=self.input_callback
        )

        # Abrindo fluxo de reprodução de áudio
        self.output_stream = self.p.open(
            format=self.format,
            channels=self.channels_out,
            rate=self.rate,
            output=True,
            frames_per_buffer=self.chunk_size,
            output_device_index=output_device_index
        )

        print("Streams de áudio inicializados com sucesso.")
        
    def add_handler(self, handler):
        """Adiciona um handler ao pipeline."""
        self.handlers.append(handler)
        self.required_handlers = len(self.handlers)

    def remove_handler(self, handler):
        """Remove um handler do pipeline."""
        if handler in self.handlers:
            self.handlers.remove(handler)
        self.required_handlers = len(self.handlers)

    def input_callback(self, in_data, frame_count, time_info, status):
        """Callback para capturar áudio do microfone."""
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        if self.channels_in == 2:  # Alterado de self.input_channels para self.channels_in
            audio_data = audio_data.reshape(-1, 2)
        self.buffer.append(audio_data)
        return (None, pyaudio.paContinue)

    def process_chunk(self, audio_data: np.ndarray) -> np.ndarray:
        """Processa um chunk de áudio através dos handlers."""
        chunk_id = str(uuid.uuid4())
        self.pipe[chunk_id].append(("input", audio_data))

        for handler in self.handlers:
            result = handler.process(chunk_id, self.pipe)
            if result is not None:
                tag, processed_data = result
                self.pipe[chunk_id].append((tag, processed_data))

        print(f"Pipe para chunk {chunk_id}: {[(tag['action'] if isinstance(tag, dict) else tag, type(data).__name__) for tag, data in self.pipe[chunk_id]]}")
        if len(self.pipe[chunk_id]) >= self.required_handlers + 1:
            final_data = self.pipe[chunk_id][-1][1]
            del self.pipe[chunk_id]
            return final_data
        return np.zeros(self.chunk_size * self.channels_out, dtype=np.int16)  # Alterado de self.output_channels para self.channels_out


    def start_streaming(self):
        """Inicia o streaming contínuo de áudio."""
        print("Iniciando streaming de áudio...")
        self.input_stream.start_stream()

        try:
            while True:
                if len(self.buffer) > 0:
                    audio_data = self.buffer.popleft()
                    # Converte mono para formato compatível se necessário
                    if self.channels_in == 1:  # Alterado de self.input_channels para self.channels_in
                        audio_data = audio_data.flatten()

                    processed_data = self.process_chunk(audio_data)
                    
                    # Converte para estéreo para saída
                    if self.channels_out == 2:  # Alterado de self.output_channels para self.channels_out
                        processed_data = np.column_stack((processed_data, processed_data)).flatten()
                    processed_bytes = processed_data.tobytes()
                    self.output_stream.write(processed_bytes)
                time.sleep(0.001)
        except KeyboardInterrupt:
            print("Streaming interrompido.")
            self.stop_streaming()


    def stop_streaming(self):
        """Finaliza os fluxos de áudio."""
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        self.p.terminate()

    def save_audio(self, frames, filename="output.wav"):
        """Salva o áudio capturado em um arquivo WAV."""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.output_channels)
            wf.setsampwidth(self.p.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
        print(f"Áudio salvo como {filename}.")