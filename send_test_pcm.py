"""Send PCM to /audio; receive live TTS chunks and play them."""
import argparse
import asyncio
import struct
import time
import wave
from pathlib import Path

import websockets

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
TTS_SAMPLE_RATE = 24000


def _try_import_sounddevice():
    try:
        import numpy as np
        import sounddevice as sd
        return sd, np
    except ImportError:
        return None, None


async def receive_tts_live(ws, out_wav: Path, max_wait_s: float, live_play: bool) -> int:
    sd, np = _try_import_sounddevice()
    chunks: list[bytes] = []
    idle_seconds = 0
    deadline = time.monotonic() + max_wait_s
    audio_stream = None

    if live_play and sd is not None:
        audio_stream = sd.OutputStream(
            samplerate=TTS_SAMPLE_RATE, channels=1, dtype="int16"
        )
        audio_stream.start()
        print(f"Live playback on ({TTS_SAMPLE_RATE} Hz PCM)...")
    elif live_play:
        print("Install sounddevice for live playback: pip install sounddevice numpy")

    print(f"Listening for TTS (up to {max_wait_s:.0f}s)...")

    while time.monotonic() < deadline:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
        except asyncio.TimeoutError:
            idle_seconds += 1
            if chunks and idle_seconds >= 5:
                break
            continue
        except websockets.exceptions.ConnectionClosed:
            break

        idle_seconds = 0
        if isinstance(msg, str):
            continue

        data = msg if isinstance(msg, bytes) else bytes(msg)
        if not data:
            continue

        chunks.append(data)
        total = sum(len(c) for c in chunks)
        print(f"Received TTS chunk: {len(data)} bytes (total {total})")

        if audio_stream is not None:
            samples = np.frombuffer(data, dtype=np.int16)
            audio_stream.write(samples)

    if audio_stream is not None:
        audio_stream.stop()
        audio_stream.close()

    if not chunks:
        server_wav = Path(__file__).resolve().parent / "last_response.wav"
        if server_wav.exists():
            print(f"No WS audio; play server file: start {server_wav}")
        return 0

    combined = b"".join(chunks)
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TTS_SAMPLE_RATE)
        wf.writeframes(combined)
    print(f"Saved -> {out_wav} ({len(combined)} bytes). Play: start {out_wav}")
    return len(combined)


async def send_pcm(
    path: Path,
    url: str,
    chunk_size: int,
    trailing_silence_s: float,
    max_wait_s: float,
    out_wav: Path,
    live_play: bool,
) -> None:
    data = path.read_bytes()
    silence = struct.pack("<h", 0) * int(SAMPLE_RATE * trailing_silence_s)
    payload = data + silence
    chunk_duration_s = chunk_size / (SAMPLE_RATE * BYTES_PER_SAMPLE)

    print(f"Sending {len(payload)} bytes to {url}")

    async with websockets.connect(url, max_size=None) as ws:
        recv_task = asyncio.create_task(
            receive_tts_live(ws, out_wav, max_wait_s, live_play)
        )

        offset = 0
        while offset < len(payload):
            await ws.send(payload[offset : offset + chunk_size])
            offset += chunk_size
            await asyncio.sleep(chunk_duration_s)

        print("Audio sent; waiting for live TTS reply...")
        await recv_task

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pcm_file", type=Path)
    parser.add_argument("--url", default="ws://127.0.0.1:8000/audio")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--trailing-silence", type=float, default=0.6)
    parser.add_argument("--max-wait", type=float, default=90.0)
    parser.add_argument("--tts-out", type=Path, default=Path("response_tts.wav"))
    parser.add_argument("--no-live-play", action="store_true")
    args = parser.parse_args()
    asyncio.run(
        send_pcm(
            args.pcm_file,
            args.url,
            args.chunk_size,
            args.trailing_silence,
            args.max_wait,
            args.tts_out,
            live_play=not args.no_live_play,
        )
    )


if __name__ == "__main__":
    main()
