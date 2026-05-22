# test_websocket.py
from fastapi import FastAPI, WebSocket
import asyncio

app = FastAPI()

@app.websocket("/test")
async def websocket_test_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Test client connected!")
    while True:
        data = await websocket.receive_text()
        print(f"Received: {data}")
        await websocket.send_text(f"Echo: {data}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)