# convert_to_raw.py
from pydub import AudioSegment
import sys

def convert_wav_to_raw(input_file, output_file, target_sr=16000):
    # 1. Load the audio file
    audio = AudioSegment.from_wav(input_file)
    
    # 2. Convert to mono and set sample rate
    audio = audio.set_channels(1).set_frame_rate(target_sr)
    
    # 3. Export as raw PCM (s16le)
    audio.export(output_file, format="raw")
    print(f"Successfully converted {input_file} to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_raw.py <input.wav> <output.raw>")
        sys.exit(1)
    convert_wav_to_raw(sys.argv[1], sys.argv[2])