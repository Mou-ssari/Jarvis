import time
import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice


class TTS:
    def __init__(self, model_path: str, config_path: str):
        print("[TTS] Loading voice model...")
        self.voice = PiperVoice.load(model_path, config_path=config_path)
        print("[TTS] Ready.")

    def speak(self, text: str):
        """Synthesize and play text immediately."""
        print(f"[TTS] Speaking: '{text}'")
        start = time.time()
        audio_chunks = []
        for chunk in self.voice.synthesize(text):
            audio_chunks.append(chunk.audio_int16_array)
        elapsed = time.time() - start

        audio = np.concatenate(audio_chunks)
        sd.play(audio, samplerate=22050)
        sd.wait()
        print(f"[TTS] Done in {elapsed:.2f}s")

    def speak_sentences(self, text: str):
        """Split text into sentences and play each as soon as it's synthesized."""
        sentences = self._split_sentences(text)
        for sentence in sentences:
            if sentence.strip():
                self.speak(sentence)

    def _split_sentences(self, text: str) -> list:
        """Simple sentence splitter on punctuation."""
        import re
        return re.split(r'(?<=[.!?])\s+', text)