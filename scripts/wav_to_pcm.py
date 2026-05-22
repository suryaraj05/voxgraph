"""Convert WAV to raw linear16 mono 16 kHz PCM for streaming tests."""
import struct
import sys
import wave
from pathlib import Path


def convert(in_path: Path, out_path: Path, target_sr: int = 16000) -> None:
    with wave.open(str(in_path), "rb") as w:
        channels = w.getnchannels()
        sr = w.getframerate()
        width = w.getsampwidth()
        frames = w.readframes(w.getnframes())

    if width != 2:
        raise ValueError(f"Expected 16-bit WAV, got {width * 8}-bit")

    samples = list(struct.unpack(f"<{len(frames) // 2}h", frames))

    if channels > 1:
        samples = [
            sum(samples[i : i + channels]) // channels
            for i in range(0, len(samples), channels)
        ]

    if sr != target_sr:
        ratio = sr / target_sr
        out_len = int(len(samples) / ratio)
        resampled = []
        for i in range(out_len):
            idx = int(i * ratio)
            resampled.append(samples[min(idx, len(samples) - 1)])
        samples = resampled

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(struct.pack(f"<{len(samples)}h", *samples))
    duration = len(samples) / target_sr
    peak = max(abs(s) for s in samples) if samples else 0
    print(f"Wrote {out_path} ({len(samples)} samples, {duration:.2f}s, peak={peak})")
    print(f"  source: {in_path} ({channels}ch, {sr}Hz, {width * 8}-bit)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/wav_to_pcm.py input.wav output.pcm")
        sys.exit(1)
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
