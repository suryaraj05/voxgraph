"""
Live microphone client for VoxGraph.

Speak into your mic, hear the AI reply through your speakers.
Designed for demos and LinkedIn-style "human moment" recordings.

Usage:
  Terminal 1:  python voxgraph.py
  Terminal 2:  pip install sounddevice numpy
               python scripts/live_mic_client.py

Press Enter to start speaking, Enter again when finished. Type q to quit.
"""
from __future__ import annotations

import argparse
import asyncio
import struct
import sys
from pathlib import Path

import websockets

# Reuse TTS receive/playback from the file-based test client
sys.path.insert(0, str(Path(__file__).resolve().parent))
from send_test_pcm import (  # noqa: E402
    BYTES_PER_SAMPLE,
    REPO_ROOT,
    SAMPLE_RATE,
    receive_tts_live,
)

try:
    import sounddevice as sd
except ImportError:
    print("Install microphone support: pip install sounddevice numpy")
    raise SystemExit(1) from None

CHUNK_FRAMES = 1024
TRAILING_SILENCE_S = 0.8
DEMO_TRAILING_SILENCE_S = 0.6


def _silence_bytes(duration_s: float) -> bytes:
    n = int(SAMPLE_RATE * duration_s)
    return struct.pack(f"<{n}h", *([0] * n))


async def _wait_enter(prompt: str) -> bool:
    """Return False if user wants to quit."""
    line = await asyncio.to_thread(input, prompt)
    if line.strip().lower() in ("q", "quit", "exit"):
        return False
    return True


async def _record_turn(
    ws: websockets.WebSocketClientProtocol,
    chunk_frames: int,
    trailing_s: float,
) -> None:
    """Stream mic PCM to the server until the user presses Enter."""
    queue: asyncio.Queue[bytes] = asyncio.Queue()
    stop = asyncio.Event()

    def audio_callback(indata, _frames, _time, status) -> None:
        if status:
            print(f"  [mic] {status}")
        queue.put_nowait(indata.tobytes())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=chunk_frames,
        callback=audio_callback,
    )

    async def pump_audio() -> None:
        stream.start()
        try:
            while not stop.is_set():
                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                await ws.send(chunk)
        finally:
            stream.stop()
            stream.close()

    pump_task = asyncio.create_task(pump_audio())

    print("  Recording... speak now, then press Enter when done.")
    await asyncio.to_thread(input)

    stop.set()
    await pump_task

    silence = _silence_bytes(trailing_s)
    await ws.send(silence)
    print("  Sent end-of-speech silence — waiting for AI reply...")


async def run_session(
    url: str,
    max_wait_s: float,
    out_wav: Path,
    once: bool,
    demo: bool,
) -> None:
    devices = sd.query_devices()
    default_in = sd.default.device[0]
    default_out = sd.default.device[1]
    print(f"Input device:  {devices[default_in]['name']}")
    print(f"Output device: {devices[default_out]['name']}")
    trailing_s = DEMO_TRAILING_SILENCE_S if demo else TRAILING_SILENCE_S
    if demo:
        print("Demo mode: low-latency playback (use DEMO_MODE=1 on server too).\n")
    print(f"Connecting to {url}\n")

    async with websockets.connect(url, max_size=None) as ws:
        print("VoxGraph live mic — press Enter to talk, q + Enter to quit.\n")

        turn = 0
        while True:
            turn += 1
            if not await _wait_enter(f"[Turn {turn}] Press Enter to speak (q to quit): "):
                break

            # Record first, then listen — avoids missing TTS while the mic task owns the socket
            await _record_turn(ws, CHUNK_FRAMES, trailing_s)
            await receive_tts_live(
                ws,
                out_wav,
                max_wait_s,
                live_play=True,
                idle_timeout_s=1.0 if demo else 5.0,
                low_latency=demo,
                min_listen_s=20.0,
            )
            print("  Reply finished.\n")

            if once:
                break

    print("Session ended.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Talk to VoxGraph with your microphone and speakers.",
    )
    parser.add_argument("--url", default="ws://127.0.0.1:8000/audio")
    parser.add_argument("--max-wait", type=float, default=90.0)
    parser.add_argument(
        "--tts-out",
        type=Path,
        default=REPO_ROOT / "response_tts.wav",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Single question then exit (good for one demo take)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Low-latency speakers + shorter trailing silence (pair with DEMO_MODE=1 on server)",
    )
    args = parser.parse_args()
    asyncio.run(
        run_session(args.url, args.max_wait, args.tts_out, args.once, args.demo)
    )


if __name__ == "__main__":
    main()
