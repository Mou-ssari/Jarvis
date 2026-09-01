import time
from stt import STT
from tts import TTS
from llm import LLM

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_LLM  = r"C:\Projects\Jarvis\models\qwen2.5-1.5b-instruct-q5_k_m.gguf"
MODEL_TTS  = r"C:\Projects\Jarvis\models\piper\en\en_US\lessac\medium\en_US-lessac-medium.onnx"
CONFIG_TTS = r"C:\Projects\Jarvis\models\piper\en\en_US\lessac\medium\en_US-lessac-medium.onnx.json"

def main():
    # Load all components once at startup
    stt = STT(model_size="base.en")
    tts = TTS(model_path=MODEL_TTS, config_path=CONFIG_TTS)
    llm = LLM(model_path=MODEL_LLM)

    print("\n" + "="*50)
    print("JARVIS is online. Press Ctrl+C to quit.")
    print("="*50 + "\n")

    tts.speak("JARVIS online. How can I help you?")

    while True:
        try:
            # Record audio until silence is detected
            audio = stt.record(silence_threshold=0.02, silence_duration=1.0)

            # Transcribe
            user_text = stt.transcribe(audio)

            if not user_text.strip():
                print("[Pipeline] No speech detected, listening again...")
                continue

            # Measure full pipeline latency
            pipeline_start = time.time()

            # Generate response
            response = llm.respond(user_text)

            # Speak response
            tts.speak_sentences(response)

            total = time.time() - pipeline_start
            print(f"[Pipeline] Total latency: {total:.2f}s\n")

        except KeyboardInterrupt:
            print("\n[Pipeline] Shutting down.")
            tts.speak("Goodbye.")
            break

if __name__ == "__main__":
    main()