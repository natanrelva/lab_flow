import pyaudio

p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f"ID {i}: {info['name']} — Input channels: {info['maxInputChannels']}, Output channels: {info['maxOutputChannels']}")
p.terminate()