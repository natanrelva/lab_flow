import pyaudio
import numpy as np

# Inicializar PyAudio
p = pyaudio.PyAudio()

# Configurar o fluxo de captura de áudio do microfone para estéreo (2 canais)
stream = p.open(format=pyaudio.paInt16,
                channels=1,  # Usando apenas 1 canal (mono)
                rate=44100,  # Taxa de amostragem
                input=True,
                frames_per_buffer=1024)

print("Capturando áudio...")

# Captura de áudio mono (1 canal)
data = np.frombuffer(stream.read(1024), dtype=np.int16)

# Converter áudio estéreo para mono (média dos canais)
if data.ndim == 2:  # Verifica se o áudio é estéreo (2 canais)
    mono_data = data.mean(axis=1).astype(np.int16)  # Média para converter em mono
else:
    mono_data = data  # Caso já seja mono, não faz nada

print("Áudio convertido para mono.")

# Fechar o fluxo de áudio
stream.stop_stream()
stream.close()
p.terminate()
