from fastapi import FastAPI, WebSocket
import uvicorn

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
    uvicorn.run(app, host="127.0.0.1", port=8000)