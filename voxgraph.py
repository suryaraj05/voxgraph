import asyncio
import base64
import json
import os
import ssl
import time
import wave
from pathlib import Path
from typing import List, Optional

import logging
import websockets

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.listen.v1.types.listen_v1results import ListenV1Results
from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict
from google.genai import client as genai_client
import uvicorn

from dotenv import load_dotenv
load_dotenv()  # loads variables from .env into os.environ

# -------------------- Configuration --------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai = genai_client.Client(api_key=GOOGLE_API_KEY)

# gemini-2.5-flash-latest is NOT a valid API id (404). Use one of these:
#   gemini-flash-latest  — auto-updates to newest Flash (what worked in your logs)
#   gemini-2.5-flash     — stable production model
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
GEMINI_MODEL_FALLBACKS = (
    "gemini-flash-latest",
    "gemini-2.5-flash",
)

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # Adam voice

# ElevenLabs stream-input returns JSON with base64 audio (not raw bytes).
# pcm_24000 is best for live playback; mp3_44100_128 works for .mp3 files.
ELEVENLABS_OUTPUT_FORMAT = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "pcm_24000")
TTS_SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", "24000"))

ELEVENLABS_URL = (
    f"wss://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream-input"
    f"?model_id=eleven_multilingual_v2&output_format={ELEVENLABS_OUTPUT_FORMAT}"
)

# -------------------- LangGraph State --------------------
class Message(TypedDict):
    role: str
    content: str

class VoxGraphState(TypedDict):
    transcript: str
    working_memory: List[Message]
    semantic_facts: List[str]
    episodic_summary: str
    llm_response: str
    needs_tool: bool

# -------------------- LangGraph Nodes --------------------
def memory_retrieval_node(state: VoxGraphState):
    # TODO: replace with real database/vector store
    semantic_facts = ["user prefers morning flights", "vegetarian"]
    episodic_summary = "last conversation was about booking a flight to Mumbai"
    return {
        "semantic_facts": semantic_facts,
        "episodic_summary": episodic_summary
    }

def llm_node(state: VoxGraphState):
    prompt = f"""
You are VoxGraph, a voice AI assistant.
User facts: {state['semantic_facts']}
Past context: {state['episodic_summary']}
User said: {state['transcript']}
"""
    response = genai.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    llm_response = response.text
    needs_tool = False  # implement tool calling later
    return {"llm_response": llm_response, "needs_tool": needs_tool}

# -------------------- Streaming LLM (yields tokens) --------------------
def _gemini_models_to_try() -> List[str]:
    ordered = [GEMINI_MODEL, *GEMINI_MODEL_FALLBACKS]
    seen = set()
    unique = []
    for name in ordered:
        if name and name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def _chunk_text(chunk) -> str:
    text = getattr(chunk, "text", None)
    if text:
        return text
    candidates = getattr(chunk, "candidates", None) or []
    if not candidates:
        return ""
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []
    return "".join(getattr(part, "text", "") or "" for part in parts)


async def llm_stream(transcript: str, semantic_facts: List[str], episodic_summary: str):
    prompt = f"""
You are VoxGraph, a voice AI assistant.
User Transcript: {transcript}
Semantic Facts: {semantic_facts}
Episodic Summary: {episodic_summary}
Reply in one or two short sentences suitable for voice output.
"""
    last_error: Optional[Exception] = None
    for model in _gemini_models_to_try():
        print(f"Calling Gemini ({model})...")
        try:
            stream = await genai.aio.models.generate_content_stream(
                model=model,
                contents=prompt,
            )
            async for chunk in stream:
                token = _chunk_text(chunk)
                if token:
                    yield token
            return
        except Exception as exc:
            last_error = exc
            print(f"Gemini ({model}) failed: {exc}")
    if last_error:
        raise last_error


async def logging_llm_stream(transcript: str, semantic_facts: List[str], episodic_summary: str):
    """Wrapper around `llm_stream` that logs each token as it is produced."""
    async for token in llm_stream(transcript, semantic_facts, episodic_summary):
        try:
            print(f"LLM token: {token}")
        except Exception:
            print("LLM token: <unprintable>")
        yield token
    print("LLM stream completed")

# -------------------- TTS Streaming (ElevenLabs) --------------------

logging.basicConfig(level=logging.INFO)
# Keep websockets quiet so transcript logs are readable
logging.getLogger("websockets").setLevel(logging.WARNING)

def _parse_elevenlabs_audio(message) -> Optional[bytes]:
    """Decode ElevenLabs stream-input JSON messages (base64 audio field)."""
    if isinstance(message, (bytes, bytearray, memoryview)):
        raw = bytes(message) if isinstance(message, memoryview) else message
        try:
            message = raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw
    if not isinstance(message, str):
        return None
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        return None
    audio_b64 = data.get("audio")
    if not audio_b64:
        return None
    return base64.b64decode(audio_b64)


def _save_tts_wav(path: Path, pcm_s16le: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_s16le)


async def tts_stream(token_stream, client_websocket: WebSocket):
    if not ELEVENLABS_API_KEY:
        print("ELEVENLABS_API_KEY not set — skipping TTS, LLM text only")
        async for _token in token_stream:
            pass
        return

    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    tts_chunks: List[bytes] = []

    async with websockets.connect(ELEVENLABS_URL, additional_headers=headers) as ws:
        await ws.send(json.dumps({
            "text": " ",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }))

        async def sender():
            async for token in token_stream:
                try:
                    print(f"Sending token to TTS: {token}")
                except Exception:
                    print("Sending token to TTS: <unprintable>")
                await ws.send(json.dumps({"text": token}))
            await ws.send(json.dumps({"text": ""}))

        async def receiver():
            async for message in ws:
                audio = _parse_elevenlabs_audio(message)
                if not audio:
                    continue
                tts_chunks.append(audio)
                print(f"TTS audio chunk (live): {len(audio)} bytes")
                try:
                    await client_websocket.send_bytes(audio)
                except (
                    websockets.exceptions.ConnectionClosedError,
                    ConnectionResetError,
                    WebSocketDisconnect,
                    RuntimeError,
                ) as exc:
                    print(f"Client closed during live TTS stream: {exc}")
                    break

        await asyncio.gather(sender(), receiver())

    if not tts_chunks:
        print("WARNING: ElevenLabs returned no audio")
        return

    combined = b"".join(tts_chunks)
    out_dir = Path(__file__).resolve().parent
    if ELEVENLABS_OUTPUT_FORMAT.startswith("pcm"):
        out_path = out_dir / "last_response.wav"
        _save_tts_wav(out_path, combined, TTS_SAMPLE_RATE)
    else:
        out_path = out_dir / "last_response.mp3"
        out_path.write_bytes(combined)
    print(f"Saved TTS -> {out_path} ({len(combined)} bytes)")
    print(f"Play with: start {out_path}")

# -------------------- FastAPI Endpoint --------------------
app = FastAPI()

SAMPLE_RATE = 16000


def _extract_transcript(message: ListenV1Results) -> str:
    alt = message.channel.alternatives[0]
    text = (alt.transcript or "").strip()
    if text:
        return text
    words = getattr(alt, "words", None) or []
    if words:
        return " ".join(w.word for w in words if getattr(w, "word", None)).strip()
    return ""


async def _response_pipeline(
    connection,
    websocket: WebSocket,
    full_transcript: str,
) -> None:
    memory = memory_retrieval_node({"transcript": full_transcript})
    semantic = memory["semantic_facts"]
    episodic = memory["episodic_summary"]
    token_gen = logging_llm_stream(full_transcript, semantic, episodic)

    if ELEVENLABS_API_KEY:
        await tts_stream(token_gen, websocket)
    else:
        async for _token in token_gen:
            pass
    print("Response pipeline finished")


async def _respond_to_utterance(
    connection,
    websocket: WebSocket,
    full_transcript: str,
) -> None:
    print(f"Sending to LLM: {full_transcript}")
    if os.getenv("STT_ONLY", "").lower() in ("1", "true", "yes"):
        print("(STT_ONLY set — skipping LLM/TTS)")
        return

    if connection.current_tts_task and not connection.current_tts_task.done():
        print("Cancelling previous response (new question)")
        connection.current_tts_task.cancel()
        try:
            await connection.current_tts_task
        except asyncio.CancelledError:
            pass

    async def run_pipeline():
        try:
            await _response_pipeline(connection, websocket, full_transcript)
        except asyncio.CancelledError:
            print("Response pipeline cancelled")
            raise
        except Exception as exc:
            print(f"LLM/TTS pipeline failed: {exc}")

    connection.last_llm_transcript = full_transcript
    connection.current_tts_task = asyncio.create_task(run_pipeline())


async def _cancel_debounce_task(connection) -> None:
    task = getattr(connection, "respond_debounce_task", None)
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def _schedule_debounced_response(
    connection,
    websocket: WebSocket,
    delay_s: float,
) -> None:
    """Call LLM only after `delay_s` seconds with no new audio (full question captured)."""

    await _cancel_debounce_task(connection)

    async def debounced():
        while True:
            silence = time.monotonic() - connection.last_audio_at
            if silence >= delay_s:
                break
            await asyncio.sleep(0.05)

        parts = list(getattr(connection, "pending_transcript_parts", []))
        full = " ".join(parts).strip()
        if not full:
            return
        if full == getattr(connection, "last_llm_transcript", None):
            return

        print(f"Debounced utterance (merged): {full}")
        await _respond_to_utterance(connection, websocket, full)

    connection.respond_debounce_task = asyncio.create_task(debounced())


async def _deepgram_keepalive_loop(connection, stop: asyncio.Event) -> None:
    """Deepgram closes if no audio/text arrives for ~10s; keep alive during LLM+TTS."""
    while not stop.is_set():
        try:
            await connection.send_keep_alive()
        except Exception:
            break
        try:
            await asyncio.wait_for(stop.wait(), timeout=4.0)
        except asyncio.TimeoutError:
            pass


async def _flush_pending_response(connection, websocket: WebSocket) -> None:
    await _cancel_debounce_task(connection)

    parts = list(getattr(connection, "pending_transcript_parts", []))
    full = " ".join(parts).strip()
    connection.pending_transcript_parts = []
    if not full:
        return
    if full == getattr(connection, "last_llm_transcript", None):
        return

    if connection.current_tts_task and not connection.current_tts_task.done():
        print("Updating LLM with fuller transcript after audio ended")
        connection.current_tts_task.cancel()
        try:
            await connection.current_tts_task
        except asyncio.CancelledError:
            pass

    print(f"Flush after audio ended: {full}")
    await _respond_to_utterance(connection, websocket, full)


@app.websocket("/audio")
async def audio_endpoint(websocket: WebSocket):
    await websocket.accept()

    if not DEEPGRAM_API_KEY:
        print("ERROR: DEEPGRAM_API_KEY not set")
        await websocket.close(code=1011, reason="Server configuration error")
        return

    deepgram = AsyncDeepgramClient(api_key=DEEPGRAM_API_KEY)

    async with deepgram.listen.v1.connect(
        model="nova-3",
        language="en",
        encoding="linear16",
        sample_rate=16000,
        channels=1,
        interim_results=True,
        endpointing=300,
        punctuate=True,
        smart_format=True,
    ) as connection:
        connection.current_tts_task = None
        connection.utterance_buffer: List[str] = []
        connection.pending_transcript_parts: List[str] = []
        connection.respond_debounce_task = None
        connection.last_llm_transcript: Optional[str] = None
        connection.last_audio_at = time.monotonic()
        debounce_s = float(os.getenv("UTTERANCE_DEBOUNCE_SEC", "2.5"))
        connection.on(EventType.OPEN, lambda _: print("Deepgram connection opened"))
        connection.on(EventType.CLOSE, lambda evt: print(f"Deepgram event CLOSE: {evt}"))
        connection.on(EventType.ERROR, lambda evt: print(f"Deepgram event ERROR: {evt}"))

        async def on_message(message):
            msg_type = getattr(message, "type", type(message).__name__)
            if msg_type != "Results":
                print(f"Deepgram event: {msg_type}")
                return

            transcript = _extract_transcript(message)
            print(
                f"STT [{msg_type}] transcript={transcript!r} "
                f"is_final={message.is_final} speech_final={message.speech_final} "
                f"from_finalize={getattr(message, 'from_finalize', None)}"
            )

            if not message.is_final:
                return

            if transcript:
                connection.utterance_buffer.append(transcript)

            end_of_utterance = message.speech_final or getattr(message, "from_finalize", False)
            if not end_of_utterance:
                return

            full_transcript = " ".join(connection.utterance_buffer).strip()
            connection.utterance_buffer.clear()
            if not full_transcript:
                print("End of utterance but buffer empty — waiting for more segments")
                return

            print(f"Utterance segment: {full_transcript}")
            connection.pending_transcript_parts.append(full_transcript)
            await _schedule_debounced_response(connection, websocket, debounce_s)

        connection.on(EventType.MESSAGE, on_message)

        listen_task = asyncio.create_task(connection.start_listening())
        keepalive_stop = asyncio.Event()
        keepalive_task = asyncio.create_task(
            _deepgram_keepalive_loop(connection, keepalive_stop)
        )

        try:
            while True:
                try:
                    audio_bytes = await websocket.receive_bytes()
                except WebSocketDisconnect:
                    print("Client websocket disconnected; finalizing Deepgram stream")
                    break

                connection.last_audio_at = time.monotonic()
                print(f"Received audio bytes from client: {len(audio_bytes)} bytes")

                if connection.pending_transcript_parts:
                    await _schedule_debounced_response(connection, websocket, debounce_s)

                try:
                    await connection.send_media(audio_bytes)
                    await asyncio.sleep(0.001)
                except (websockets.exceptions.ConnectionClosedError, ssl.SSLEOFError, OSError) as exc:
                    print(f"Deepgram send_media failed: {exc}")
                    break
        except Exception as exc:
            print(f"Unhandled audio_endpoint error: {exc}")
        finally:
            keepalive_stop.set()
            keepalive_task.cancel()
            try:
                await keepalive_task
            except asyncio.CancelledError:
                pass

            try:
                await connection.send_finalize()
                await connection.send_close_stream()
            except Exception as exc:
                print(f"Deepgram finalize failed: {exc}")

            # Let start_listening drain final Results (from_finalize) before cancelling
            try:
                await asyncio.wait_for(listen_task, timeout=5.0)
            except asyncio.TimeoutError:
                print("Timed out waiting for Deepgram to close; cancelling listener")
                listen_task.cancel()
                try:
                    await listen_task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass

            await _flush_pending_response(connection, websocket)

            if connection.current_tts_task and not connection.current_tts_task.done():
                try:
                    await asyncio.wait_for(connection.current_tts_task, timeout=90.0)
                except asyncio.TimeoutError:
                    print("LLM/TTS still running after 90s; cancelling")
                    connection.current_tts_task.cancel()
                except asyncio.CancelledError:
                    pass
            try:
                await websocket.close()
            except Exception:
                pass

# -------------------- Run Server --------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)