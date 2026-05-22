"""Send raw PCM (linear16, mono, 16 kHz) to the VoxGraph /audio websocket."""
import argparse
import asyncio
import struct
from pathlib import Path

import websockets

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # linear16


async def send_pcm(
    path: Path,
    url: str,
    chunk_size: int,
    trailing_silence_s: float,
    post_send_wait_s: float,
) -> None:
    data = path.read_bytes()
    silence = struct.pack("<h", 0) * int(SAMPLE_RATE * trailing_silence_s)
    payload = data + silence
    chunk_duration_s = chunk_size / (SAMPLE_RATE * BYTES_PER_SAMPLE)

    print(
        f"Sending {len(data)} bytes + {len(silence)} bytes silence "
        f"from {path} to {url} (~{chunk_duration_s:.3f}s per chunk)"
    )

    async with websockets.connect(url) as ws:
        offset = 0
        while offset < len(payload):
            chunk = payload[offset : offset + chunk_size]
            await ws.send(chunk)
            offset += len(chunk)
            await asyncio.sleep(chunk_duration_s)

        # Keep socket open so Deepgram can endpoint and return speech_final / finalize
        print(f"Audio sent; waiting {post_send_wait_s}s before closing...")
        await asyncio.sleep(post_send_wait_s)

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pcm_file", type=Path)
    parser.add_argument("--url", default="ws://127.0.0.1:8000/audio")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument(
        "--trailing-silence",
        type=float,
        default=0.6,
        help="Seconds of silence appended after file (helps speech_final)",
    )
    parser.add_argument(
        "--post-send-wait",
        type=float,
        default=8.0,
        help="Seconds to keep websocket open after last chunk (for LLM+TTS reply)",
    )
    args = parser.parse_args()
    asyncio.run(
        send_pcm(
            args.pcm_file,
            args.url,
            args.chunk_size,
            args.trailing_silence,
            args.post_send_wait,
        )
    )


if __name__ == "__main__":
    main()
