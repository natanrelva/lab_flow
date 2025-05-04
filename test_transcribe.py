# test_transcribe.py
import uuid
import pickle
import tempfile
import numpy as np
import scipy.io.wavfile as wavfile
from handlers import DecodeHandler, SerializeHandler, TranscribeHandler

# 1) Substitua 'sample_pt.wav' por um WAV real de fala em português
data_rate, data = wavfile.read('sample_pt_converted.wav')
pipe_id = str(uuid.uuid4())
pipe = {pipe_id: [("input", data.astype(np.int16))]}

# 2) Decode
dec = DecodeHandler()
tag1, decoded = dec.process(pipe_id, pipe)
pipe[pipe_id].append((tag1, decoded))

# 3) Serialize
ser = SerializeHandler()
tag2, serialized = ser.process(pipe_id, pipe)
pipe[pipe_id].append((tag2, serialized))

# 4) Transcribe
trans = TranscribeHandler()
tag3, text = trans.process(pipe_id, pipe)
print("Transcribed (from WAV):", text)