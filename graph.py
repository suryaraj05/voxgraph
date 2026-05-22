import asyncio
from typing import List

from deepgram import DeepgramClient
from deepgram.core.events import EventType
from fastapi import WebSocket
from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict
from google.genai import client as genai_client
import websockets
import json

import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai = genai_client.Client(api_key=GOOGLE_API_KEY)
deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
eleven_api_key = os.getenv("ELEVENLABS_API_KEY")

current_tts_task = None

# @app.websocket("/audio")
# async def audio_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     deepgram = DeepgramClient()
#     with deepgram.listen.v1.connect(model="nova-3") as connection:
#         connection.on(EventType.OPEN, lambda _: print("Opened"))
#         connection.on(EventType.MESSAGE, on_message)
#         while True:
#             data = await websocket.receive_bytes()  
#             if current_tts_task and not current_tts_task.done():
#                 current_tts_task.cancel()
#             connection.send(data)  # to Deepgram


@app.websocket("/audio")
async def audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    deepgram = DeepgramClient()
    
    with deepgram.listen.v1.connect(model="nova-3") as connection:
        connection.on(EventType.OPEN, lambda _: print("Opened"))
        connection.current_tts_task = None
        
        def on_message(message):
            alternative = message.channel.alternatives[0]
            transcript = alternative.transcript
            if not transcript:
                return
            is_final = message.is_final
            speech_final = message.speech_final
            if is_final and speech_final:
                memory = memory_retrieval_node({"transcript":transcript})
                semantic_facts = memory.get("semantic_facts")
                episodic_summary = memory.get("episodic_summary")

                token_stream = llm_stream(transcript, semantic_facts, episodic_summary)
                connection.current_tts_task =  asyncio.create_task(tts_stream(token_stream, websocket))
                
        connection.on(EventType.MESSAGE, on_message)

        # also set connection.current_tts_task = None here
        while True:
            data = await websocket.receive_bytes()
            # cancellation check here using connection.current_tts_task
            if connection.current_tts_task and not connection.current_tts_task.done():
                connection.current_tts_task.cancel()
            connection.send(data)




class Message(TypedDict):
    role: str      # "user" or "assistant"
    content: str

# STEP 1: create the graph with your state
class VoxGraphState(TypedDict):
    transcript: str              # raw STT output
    working_memory: List[Message] # conversation history
    semantic_facts: List[str]    # persistent user facts
    episodic_summary: str        # summary of past conversations
    llm_response: str            # what LLM generated
    needs_tool: bool             # conditional edge decision


graph = StateGraph(VoxGraphState)

# STEP 2: add your nodes

def memory_retrieval_node(state: VoxGraphState):
    # fetch semantic facts (hardcoded for now)
    semantic_facts: List[str] = [
        "hello world",
        "good morning",
        "good night"
    ]
    # fetch episodic summary (hardcoded for now)
    episodic_summary: str = "this is the episodic summary of the voxgraph"
    # return updated state
    return {
        "semantic_facts" : semantic_facts,
        "episodic_summary" : episodic_summary
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

    llm_response: str = response.text
    needs_tool: bool = False

    return {
         "llm_response": llm_response ,
         "needs_tool":  needs_tool
    }



voice_id = "JBFqnCBsd6RMkjVDRZzb"
url = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id=eleven_multilingual_v2?api_key={eleven_api_key}"

async def tts_stream(llm_response_stream, websocket):
    # STEP 1: connect to ElevenLabs stream-input WebSocket
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({
            "text": " ",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
        }))

        
        async def sender():
            for chunk in llm_response_stream:
                await ws.send(json.dumps({
                    "text" : chunk
                }))
            await ws.send(json.dumps({
                "text": "",
            }))

        async def receiver():
            async for audio in ws:
                await websocket.send_bytes(audio)

        await asyncio.gather(sender(), receiver())


async def llm_stream(transcript: str, semantic_facts: List[str], episodic_summary: str):
    # Build the prompt
    # Configure Gemini (you already have client outside)
    # Call model with stream=True
    prompt = f"""
You are VoxGraph, a voice AI assistant.
User Transcript: {transcript},
Semantic Facts: {semantic_facts},
Episodic Summary: {episodic_summary}
"""
    for token in genai.models.generate_content_stream(
        model="gemini-2.5-flash-latest",
        contents=prompt,
    ):
        yield token.text
    
    

# def on_message(message):
#     alternative = message.channel.alternatives[0]
#     transcript = alternative.transcript
#     if not transcript:
#         return
#     is_final = message.is_final
#     speech_final = message.speech_final
#     if is_final and speech_final:
#         memory = memory_retrieval_node({"transcript":transcript})
#         semantic_facts = memory.get("semantic_facts")
#         episodic_summary = memory.get("episodic_summary")

#         token_stream = llm_stream(transcript, semantic_facts, episodic_summary)
#         current_tts_task =  asyncio.create_task(tts_stream(token_stream, websocket))



graph.add_node("memory_retrieval", memory_retrieval_node)
graph.add_node("llm", llm_node)

# STEP 3: add edges
#         START -> memory_retrieval
#         memory_retrieval -> llm
#         llm -> END (for now, no tools yet)

graph.add_edge(START, "memory_retrieval")
graph.add_edge("memory_retrieval", "llm")
graph.add_edge("llm", END)

# STEP 4: compile the graph
app = graph.compile()

app.invoke({"transcript": transcript})