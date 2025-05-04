# test_decode.py
import uuid
import time
import numpy as np
from handlers import DecodeHandler

# Simula um chunk de áudio int16 (ex.: 1 segundo mono)
audio_int16 = np.random.randint(-32768,32767,size=44100,dtype=np.int16)

# Prepara o pipe
test_id = str(uuid.uuid4())
pipe = {test_id: [("input", audio_int16)]}

# Executa o handler
decode = DecodeHandler()
tag, out = decode.process(test_id, pipe)
print("Tag:", tag)
print("Output dtype:", out.dtype, "min/max:", out.min(), out.max())