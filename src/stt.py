import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


class STT:
    def __init__(self, model_size: str = "base.en", device: str = "cpu"):
        print(f"[STT] Loading {model_size}...")
        self.model = WhisperModel(model_size, device=device, compute_type="int8")
        self.sample_rate = 16000
        print("[STT] Ready.")

    def record(self, max_duration: float = 10.0, silence_threshold: float = 0.01,
               silence_duration: float = 1.5) -> np.ndarray:
        """
        Record until silence is detected or max_duration is reached.
        silence_threshold: RMS below this = silence
        silence_duration:  seconds of silence before stopping
        """
        print("[STT] Listening...")
        block_size = int(self.sample_rate * 0.1)  # 100ms blocks
        max_blocks = int(max_duration / 0.1)
        silence_blocks = int(silence_duration / 0.1)

        recorded = []
        silent_count = 0
        speech_started = False

        with sd.InputStream(samplerate=self.sample_rate, channels=1,
                            dtype="float32", blocksize=block_size) as stream:
            for _ in range(max_blocks):
                block, _ = stream.read(block_size)
                block = block.flatten()
                rms = np.sqrt(np.mean(block ** 2))

                if rms > silence_threshold:
                    speech_started = True
                    silent_count = 0
                    recorded.append(block)
                elif speech_started:
                    recorded.append(block)
                    silent_count += 1
                    if silent_count >= silence_blocks:
                        print("[STT] Silence detected, processing...")
                        break

        if not recorded:
            return np.array([], dtype=np.float32)

        return np.concatenate(recorded)

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio array to text."""
        if len(audio) == 0:
            return ""
        start = time.time()
        segments, _ = self.model.transcribe(
            audio,
            language="en",
            vad_filter=True
        )
        text = " ".join([s.text.strip() for s in segments])
        elapsed = time.time() - start
        print(f"[STT] Transcribed in {elapsed:.2f}s: '{text}'")
        return text