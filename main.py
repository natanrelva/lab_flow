from streaming_audio import RealTimeAudioStream

def main():
    # Inicializa o módulo de streaming de áudio
    audio_stream = RealTimeAudioStream(rate=44100, chunk_size=1024, channels=1)

    # Captura e reproduz áudio por 10 segundos
    frames = audio_stream.capture_and_play(duration=10)

    # Opcional: Salva o áudio em um arquivo WAV
    audio_stream.save_audio(frames, filename="output.wav")

if __name__ == "__main__":
    main()
