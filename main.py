import pyaudio
from real_time_audio_stream import RealTimeAudioStream
from handlers import (
    DecodeHandler,
    SerializeHandler,
    TranscribeHandler,
    TranslateHandler,
    SerializeTextHandler,
    EncodeHandler
)

def main():
    mic_index = 2    # Microfone (MacBook Pro)
    out_index = 1    # BlackHole 2ch (saída virtual estéreo)

    stream = RealTimeAudioStream(
        rate=44100,
        chunk=1024,
        in_ch=1,
        out_ch=2,
        fmt=pyaudio.paInt16,
        in_dev=2,    # microfone do MacBook (ID 2)
        out_dev=1    # BlackHole 2ch (ID 1)
    )

    stream.add_handler(DecodeHandler())
    stream.add_handler(SerializeHandler())
    stream.add_handler(TranscribeHandler())
    stream.add_handler(TranslateHandler())
    stream.add_handler(SerializeTextHandler())
    stream.add_handler(EncodeHandler())

    stream.start()

if __name__ == "__main__":
    main()
