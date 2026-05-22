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

    out_path.write_bytes(struct.pack(f"<{len(samples)}h", *samples))
    duration = len(samples) / target_sr
    peak = max(abs(s) for s in samples) if samples else 0
    print(f"Wrote {out_path} ({len(samples)} samples, {duration:.2f}s, peak={peak})")
    print(f"  source: {in_path} ({channels}ch, {sr}Hz, {width * 8}-bit)")


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "testing.wav")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "testing_5s_fixed.pcm")
    convert(src, dst)
