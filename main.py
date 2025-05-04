import pyaudio
from real_time_audio_stream import RealTimeAudioStream
from handlers import DecodeHandler, SerializeHandler, TranscribeHandler, TranslateHandler, SerializeTextHandler, EncodeHandler

def main():
    # Inicializa o módulo de streaming de áudio
    audio_stream = RealTimeAudioStream(
        rate=44100,
        chunk_size=44100,  # 1 segundo para transcrição
        channels_in=1,     # Mono para captura (microfone)
        channels_out=2,    # Estéreo para BlackHole/Multi-Output
        format=pyaudio.paInt16,
        input_device_index=None,  # Microfone padrão
        output_device_index=3    # Ajuste para o índice do Multi-Output Device
    )

    # Adiciona handlers para processamento em fluxo
    audio_stream.add_handler(DecodeHandler())
    audio_stream.add_handler(SerializeHandler())
    audio_stream.add_handler(TranscribeHandler())
    audio_stream.add_handler(TranslateHandler())
    audio_stream.add_handler(SerializeTextHandler())
    audio_stream.add_handler(EncodeHandler())

    # Inicia o streaming contínuo
    audio_stream.start_streaming()

if __name__ == "__main__":
    main()
