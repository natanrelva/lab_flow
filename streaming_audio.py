import pyaudio
import numpy as np
import wave

class RealTimeAudioStream:
    def __init__(self, rate=44100, chunk_size=1024, channels=1, format=pyaudio.paInt16):
        # Inicializando parâmetros de áudio
        self.rate = rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = format
        self.p = pyaudio.PyAudio()
        
        # Abrindo fluxo de captura de áudio
        self.input_stream = self.p.open(format=self.format,
                                        channels=self.channels,
                                        rate=self.rate,
                                        input=True,
                                        frames_per_buffer=self.chunk_size)
        
        # Abrindo fluxo de reprodução de áudio
        self.output_stream = self.p.open(format=self.format,
                                         channels=self.channels,
                                         rate=self.rate,
                                         output=True,
                                         frames_per_buffer=self.chunk_size)

        print("Streams de áudio inicializados com sucesso.")

    def capture_and_play(self, duration=10):
        """
        Captura e reproduz áudio em tempo real por um dado período de tempo.
        """
        print("Iniciando captura e reprodução de áudio...")
        
        # Lista para armazenar os frames capturados
        frames = []

        # Captura e reprodução de áudio
        for _ in range(0, int(self.rate / self.chunk_size * duration)):
            # Captura dados do microfone
            data = self.input_stream.read(self.chunk_size)
            frames.append(data)

            # Reproduz os dados capturados em tempo real
            self.output_stream.write(data)

        print("Captura e reprodução concluídas.")

        # Finaliza o fluxo de captura e reprodução
        self.input_stream.stop_stream()
        self.input_stream.close()
        self.output_stream.stop_stream()
        self.output_stream.close()

        self.p.terminate()

        return frames  # Retorna os frames capturados para armazenar ou processar posteriormente

    def save_audio(self, frames, filename="output.wav"):
        """
        Salva o áudio capturado em um arquivo WAV.
        """
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
        
        print(f"Áudio salvo como {filename}.")
