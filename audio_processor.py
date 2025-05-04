import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize


class AudioProcessor:
    def process(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Método para processar o áudio. Esse método pode ser sobrescrito para adicionar novos tratamentos no pipeline.
        :param audio_data: Dados de áudio em formato numpy.
        :return: Dados de áudio processados.
        """
        raise NotImplementedError("O método 'process' precisa ser implementado.")


class NormalizationProcessor(AudioProcessor):
    def process(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Exemplo de processamento: normaliza o áudio.
        :param audio_data: Dados de áudio em formato numpy.
        :return: Dados de áudio normalizados.
        """
        # Convertendo para pydub AudioSegment
        audio_segment = AudioSegment(
            audio_data.tobytes(),
            frame_rate=44100,
            sample_width=2,
            channels=1
        )
        normalized_audio = normalize(audio_segment)
        # Convertendo de volta para numpy array
        return np.array(normalized_audio.get_array_of_samples())
