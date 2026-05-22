import wave
import struct
import math

with open("test_sine.raw", "wb") as f:
    for i in range(16000):  # 1 second of 440Hz sine
        val = int(32767 * math.sin(2 * math.pi * 440 * i / 16000))
        f.write(struct.pack('<h', val))