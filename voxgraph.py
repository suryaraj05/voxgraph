import asyncio
import json
import os
import ssl
from typing import List

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

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # Adam voice

ELEVENLABS_URL = (
    f"wss://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream-input"
    f"?model_id=eleven_multilingual_v2&api_key={ELEVENLABS_API_KEY}"
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
        model="gemini-2.5-flash-latest",
        contents=prompt,
    )
    llm_response = response.text
    needs_tool = False  # implement tool calling later
    return {"llm_response": llm_response, "needs_tool": needs_tool}

# -------------------- Streaming LLM (yields tokens) --------------------
async def llm_stream(transcript: str, semantic_facts: List[str], episodic_summary: str):
    prompt = f"""
You are VoxGraph, a voice AI assistant.
User Transcript: {transcript}
Semantic Facts: {semantic_facts}
Episodic Summary: {episodic_summary}
"""
    for chunk in genai.models.generate_content_stream(
        model="gemini-2.5-flash-latest",
        contents=prompt,
    ):
        yield chunk.text


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

async def tts_stream(token_stream, client_websocket: WebSocket):
    async with websockets.connect(ELEVENLABS_URL) as ws:
        # Initialize stream
        await ws.send(json.dumps({
            "text": " ",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
        }))

        async def sender():
            async for token in token_stream:
                try:
                    print(f"Sending token to TTS: {token}")
                except Exception:
                    print("Sending token to TTS: <unprintable>")
                await ws.send(json.dumps({"text": token}))
            await ws.send(json.dumps({"text": ""}))  # end signal

        async def receiver():
            async for audio in ws:
                try:
                    print(f"TTS audio chunk size: {len(audio)} bytes")
                except Exception:
                    print("TTS audio chunk received")
                try:
                    await client_websocket.send_bytes(audio)
                except (websockets.exceptions.ConnectionClosedError, ConnectionResetError, WebSocketDisconnect):
                    print("Client websocket closed while sending TTS audio")
                    break

        await asyncio.gather(sender(), receiver())

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


async def _respond_to_utterance(
    connection,
    websocket: WebSocket,
    full_transcript: str,
) -> None:
    print(f"Utterance complete: {full_transcript}")
    memory = memory_retrieval_node({"transcript": full_transcript})
    semantic = memory["semantic_facts"]
    episodic = memory["episodic_summary"]
    token_gen = llm_stream(full_transcript, semantic, episodic)
    connection.current_tts_task = asyncio.create_task(tts_stream(token_gen, websocket))


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

            await _respond_to_utterance(connection, websocket, full_transcript)

        connection.on(EventType.MESSAGE, on_message)

        listen_task = asyncio.create_task(connection.start_listening())

        try:
            while True:
                try:
                    audio_bytes = await websocket.receive_bytes()
                except WebSocketDisconnect:
                    print("Client websocket disconnected; finalizing Deepgram stream")
                    break

                print(f"Received audio bytes from client: {len(audio_bytes)} bytes")

                if connection.current_tts_task and not connection.current_tts_task.done():
                    print("Cancelling current TTS task due to new audio (barge-in)")
                    connection.current_tts_task.cancel()
                    try:
                        await connection.current_tts_task
                    except asyncio.CancelledError:
                        pass

                try:
                    await connection.send_media(audio_bytes)
                    await asyncio.sleep(0.001)
                except (websockets.exceptions.ConnectionClosedError, ssl.SSLEOFError, OSError) as exc:
                    print(f"Deepgram send_media failed: {exc}")
                    break
        except Exception as exc:
            print(f"Unhandled audio_endpoint error: {exc}")
        finally:
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

            if connection.current_tts_task and not connection.current_tts_task.done():
                connection.current_tts_task.cancel()
            try:
                await websocket.close()
            except Exception:
                pass

# -------------------- Run Server --------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)