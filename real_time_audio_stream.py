import pyaudio
import numpy as np
import wave
import uuid
import time
from collections import deque, defaultdict

class RealTimeAudioStream:
    def __init__(self,
                 rate=44100,
                 chunk=44100,
                 in_ch=1,
                 out_ch=2,
                 fmt=pyaudio.paInt16,
                 in_dev=None,
                 out_dev=None):
        self.rate = rate
        self.chunk_size = chunk
        self.channels_in = in_ch
        self.channels_out = out_ch
        self.format = fmt
        self.p = pyaudio.PyAudio()
        self.handlers = []
        self.pipe = defaultdict(list)
        self.buffer = deque(maxlen=20)

        print(f"[RTAS] Config: rate={rate}, chunk={chunk}, in_ch={in_ch}, in_dev={in_dev}, out_ch={out_ch}, out_dev={out_dev}")

        # input stream
        self.input_stream = self.p.open(
            format=self.format,
            channels=self.channels_in,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=in_dev,
            stream_callback=self._callback
        )

        # output stream
        self.output_stream = self.p.open(
            format=self.format,
            channels=self.channels_out,
            rate=self.rate,
            output=True,
            frames_per_buffer=self.chunk_size,
            output_device_index=out_dev
        )

        print("Streams de áudio inicializados com sucesso.")

    def add_handler(self, handler):
        self.handlers.append(handler)

    def _callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        print(f"[Callback] Capturou {len(audio_data)} amostras")
        if self.channels_in == 2:
            audio_data = audio_data.reshape(-1, 2)
        self.buffer.append(audio_data)
        return (None, pyaudio.paContinue)

    def process_chunk(self, audio_data: np.ndarray) -> np.ndarray:
        chunk_id = str(uuid.uuid4())
        self.pipe[chunk_id].append(("input", audio_data))

        for handler in self.handlers:
            result = handler.process(chunk_id, self.pipe)
            if result:
                tag, processed = result
                self.pipe[chunk_id].append((tag, processed))
                print(f"[{handler.__class__.__name__}] chunk {chunk_id} → {tag['action']}")

        # return last encode or silence
        for tag, data in reversed(self.pipe[chunk_id]):
            if isinstance(tag, dict) and tag.get("action") == "encode":
                del self.pipe[chunk_id]
                return data

        del self.pipe[chunk_id]
        return np.zeros(self.chunk_size * self.channels_out, dtype=np.int16)

    def start(self):
        print("Iniciando streaming de áudio...")
        self.input_stream.start_stream()

        try:
            while True:
                if self.buffer:
                    audio_data = self.buffer.popleft()
                    if self.channels_in == 1:
                        audio_data = audio_data.flatten()
                    processed = self.process_chunk(audio_data)
                    self.output_stream.write(processed.tobytes())
                time.sleep(0.005)
        except KeyboardInterrupt:
            print("Streaming interrompido.")
        finally:
            self.stop()

    def stop(self):
        self.input_stream.stop_stream()
        self.input_stream.close()
        self.output_stream.stop_stream()
        self.output_stream.close()
        self.p.terminate()
        print("Streams finalizados.")

    def save_audio(self, frames, filename="output.wav"):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels_out)
            wf.setsampwidth(self.p.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
        print(f"Áudio salvo como {filename}.")
