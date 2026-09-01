import time
import os

from stt import STT
from tts import TTS
from llm import LLM
from wakeup import WakeWord

# Paths
MODEL_LLM  = r"C:\Projects\Jarvis\models\qwen2.5-1.5b-instruct-q5_k_m.gguf"
MODEL_TTS  = r"C:\Projects\Jarvis\models\piper\en\en_US\lessac\medium\en_US-lessac-medium.onnx"
CONFIG_TTS = r"C:\Projects\Jarvis\models\piper\en\en_US\lessac\medium\en_US-lessac-medium.onnx.json"


DISMISSAL_PHRASES = [
    "goodbye", "goodbye jarvis", "bye", "bye jarvis",
    "that's all", "that's all jarvis", "thank you jarvis",
    "go to sleep", "sleep"
]

def is_dismissal(text: str) -> bool:
    return any(phrase in text.lower().strip() for phrase in DISMISSAL_PHRASES)


def main():
    # Load all components once at startup
    stt = STT(model_size="base.en")
    tts = TTS(model_path=MODEL_TTS, config_path=CONFIG_TTS)
    llm = LLM(model_path=MODEL_LLM)
    wake = WakeWord(wake_word = "hey_jarvis", threshold = 0.5)

    print("\n" + "="*50)
    print("JARVIS is online. Press Ctrl+C to quit.")
    print("="*50 + "\n")

    # stt.calibrate(duration = 1.5) #calibrate the microphone during startup
    stt.calibrate(duration=2.0)


    tts.speak("JARVIS online. How can I help you?")
    time.sleep(1)  # slight pause before starting the loop

    while True:
        try:
            wake.listen() # wait for the wake word

            tts.speak("Yes?") # ACK the wake word
            time.sleep(0.5)  # slight pause before recording
            while True:
                # Record audio until silence is detected
                audio = stt.record(silence_duration= 2.0)

                # Transcribe
                user_text = stt.transcribe(audio)

                if not user_text.strip():
                    print("[Pipeline] No speech detected, listening again...")
                    tts.speak("I didn't catch that. Go ahead.")
                    time.sleep(0.3)
                    continue

                print(f"[Pipeline] Processing: '{user_text}'")

                # Check for dismissal before sending to LLM
                if is_dismissal(user_text):
                    tts.speak("Goodbye. Call me when you need me.")
                    time.sleep(1.0)
                    print("[Pipeline] Returned to sleep.\n")
                    break  # Exit conversation loop, back to wake word


                pipeline_start = time.time() # Measure full pipeline latency
                response = llm.respond(user_text) # Generate response
                tts.speak_sentences(response) # Speak response

                total = time.time() - pipeline_start
                print(f"[Pipeline] Total latency: {total:.2f}s\n")
                time.sleep(0.8)  # Let response audio clear before listening again

        except KeyboardInterrupt:
            print("\n[Pipeline] Shutting down.")
            tts.speak("Shutting down.")
            break

if __name__ == "__main__":
    main()