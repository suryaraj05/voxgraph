"""Analyze PCM file and test Deepgram file transcription."""
import os
import struct
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def analyze_pcm(path: Path, sample_rate: int = 16000) -> None:
    data = path.read_bytes()
    n_samples = len(data) // 2
    samples = struct.unpack(f"<{n_samples}h", data[: n_samples * 2])
    peak = max(abs(s) for s in samples) if samples else 0
    rms = (sum(s * s for s in samples) / len(samples)) ** 0.5 if samples else 0
    duration = n_samples / sample_rate
    nonzero = sum(1 for s in samples if abs(s) > 500)
    print(f"File: {path}")
    print(f"  bytes={len(data)} samples={n_samples} duration@{sample_rate}Hz={duration:.2f}s")
    print(f"  peak={peak} rms={rms:.1f} active_samples(>500)={nonzero}/{n_samples}")


def transcribe_file(path: Path) -> None:
    from deepgram import DeepgramClient

    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        print("DEEPGRAM_API_KEY not set")
        return

    client = DeepgramClient(api_key=api_key)
    with open(path, "rb") as f:
        audio = f.read()

    for label, kwargs in [
        ("linear16 16k mono", {"encoding": "linear16", "sample_rate": 16000, "channels": 1}),
        ("linear16 48k mono", {"encoding": "linear16", "sample_rate": 48000, "channels": 1}),
        ("no encoding hint", {}),
    ]:
        try:
            resp = client.listen.v1.media.transcribe_file(
                request=audio,
                model="nova-3",
                language="en",
                **kwargs,
            )
            t = resp.results.channels[0].alternatives[0].transcript
            print(f"  REST [{label}]: {t!r}")
        except Exception as e:
            print(f"  REST [{label}]: ERROR {e}")


if __name__ == "__main__":
    pcm = Path(sys.argv[1] if len(sys.argv) > 1 else "testing_5s.pcm")
    analyze_pcm(pcm)
    transcribe_file(pcm)
