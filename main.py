import asyncio
import websockets
import random
import time

try:
    async def send(websocket, path):
        while True:
            await websocket.send("Hello World!")
            await asyncio.sleep(1)
            
    start_server = websockets.serve(send, "", 8000)

    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

except KeyboardInterrupt:
    print("\nSERVER: Stopped")

    



