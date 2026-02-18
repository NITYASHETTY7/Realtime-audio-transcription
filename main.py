import asyncio
import os
import json
import time
import websockets
import pyaudio
from dotenv import load_dotenv

from intelligence import (
    analyze_conversation,
    search_manuals,
    generate_solution_card
)

load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

RATE = 16000
CHUNK = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16
MIC_DEVICE_INDEX = 1
transcript_buffer = []
MAX_BUFFER_LINES = 20
last_call_time = 0
MIN_WORDS = 12
COOLDOWN_SECONDS = 15

def add_to_buffer(text):
    transcript_buffer.append(text)
    if len(transcript_buffer) > MAX_BUFFER_LINES:
        transcript_buffer.pop(0)

def handle_transcript(full_text):
    global last_call_time
    if len(full_text.split()) < MIN_WORDS:
        print("Not enough context yet.")
        return
    if time.time() - last_call_time < COOLDOWN_SECONDS:
        print("⏳ Cooldown active.")
        return
    last_call_time = time.time()

    print("\nAnalyzing conversation...\n")
    analysis = analyze_conversation(full_text)
    if not analysis:
        print(" Analysis failed. Waiting for next speech.")
        return
    print("Sentiment:", analysis["sentiment"])
    print("Category:", analysis["category"])
    print("Search Query:", analysis["search_query"])
    print("\nSearching manuals...\n")
    results = search_manuals(analysis["search_query"])
    if not results:
        print(" No relevant manual sections found.")
        return

    print("Generating solution card...\n")
    solution = generate_solution_card(
        analysis["search_query"],
        results
    )
    print("\n==============================")
    print("       SOLUTION CARD")
    print("==============================\n")
    print(solution)
    print("\n================================\n")


async def main():
    url = (
        "wss://api.deepgram.com/v1/listen"
        "?model=nova-2"
        "&language=en-US"
        "&encoding=linear16"
        "&sample_rate=16000"
        "&channels=1"
        "&interim_results=true"
    )
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/raw"
    }
    async with websockets.connect(
        url,
        additional_headers=headers,
        ping_interval=20,
        ping_timeout=20
    ) as ws:
        async def receive():
            try:
                async for message in ws:
                    data = json.loads(message)
                    if "channel" in data:
                        transcript = data["channel"]["alternatives"][0]["transcript"]
                        if transcript and data.get("is_final"):
                            print("\nFinal Transcript:", transcript)
                            add_to_buffer(transcript)
                            print(" Speech pause detected")
                            full_text = " ".join(transcript_buffer)
                            handle_transcript(full_text)
            except Exception as e:
                print("Deepgram connection closed:", e)

        async def send_audio():
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=MIC_DEVICE_INDEX,
                frames_per_buffer=CHUNK,
            )
            print("Listening.......\n")

            try:
                while True:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    await ws.send(data)
            except KeyboardInterrupt:
                pass
            except Exception as e:
                print(" Audio send error:", e)
            finally:
                stream.stop_stream()
                stream.close()
                audio.terminate()
        await asyncio.gather(send_audio(), receive())


if __name__ == "__main__":
    asyncio.run(main())
