import numpy as np
import sounddevice as sd
from openwakeword.model import Model

class WakeWord:
    def __init__(self, wake_word: str = "hey_jarvis", threshold: float = 0.5):
        print(f"[Wake] Loading ' {wake_word}' model...")
        self.model = Model(wakeword_models=[wake_word], inference_framework = "onnx")
        self.wake_word = wake_word
        self.threshold = threshold
        self.sample_rate = 16000
        self.chunk_size = 1280 # 80 ms chunks at 16kHz
        print("[Wake] Ready!")

    def listen(self) -> bool:
        """
        Listen for the wake word in real-time audio.
        Blocks until wake word is detected, then returns True.
        """
        print(f"[wake] Waiting for ¨{self.wake_word}'...")
        self.model.reset()

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="int16", blocksize=self.chunk_size) as stream:
            while True:
                chunk, _ = stream.read(self.chunk_size)
                chunk = chunk.flatten()

                predictions = self.model.predict(chunk)
                score = predictions.get(self.wake_word, 0.0)

                if score >= self.threshold:
                    print(f"[wake] Detected '{self.wake_word}' (score: {score:.2f})")
                    return True
                    