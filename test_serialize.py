import uuid
import pickle
import numpy as np
from handlers import DecodeHandler, SerializeHandler

# Primeiro decodifica
import uuid; test_id = str(uuid.uuid4())
audio = np.random.randint(-32768,32767,size=44100,dtype=np.int16)
pipe = {test_id: [("input", audio)]}
dec = DecodeHandler(); tag1, decoded = dec.process(test_id, pipe)
pipe[test_id].append((tag1, decoded))

# Agora serializa
ser = SerializeHandler()
tag2, data_bytes = ser.process(test_id, pipe)
print("Tag:", tag2)
print("Bytes length:", len(data_bytes))

# Verifica desserialização
restored = pickle.loads(data_bytes)
print("Restored dtype:", restored.dtype)