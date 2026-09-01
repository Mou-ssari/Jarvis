import time
from llama_cpp import Llama


class LLM:
    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = 8):
        print("[LLM] Loading model — this may take a moment...")
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False
        )
        self.system_prompt = (
            "You are JARVIS, a concise and intelligent personal AI assistant. "
            "Keep responses short — 1 to 3 sentences maximum. "
            "Be direct and helpful."
        )
        print("[LLM] Ready.")

    def respond(self, user_input: str) -> str:
        """Generate a response to user input."""
        start = time.time()
        response = self.model.create_chat_completion(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input}
            ],
            max_tokens=150,
            temperature=0.7
        )
        elapsed = time.time() - start
        text = response["choices"][0]["message"]["content"].strip()
        tokens = response["usage"]["completion_tokens"]
        print(f"[LLM] Response in {elapsed:.2f}s ({tokens} tokens): '{text}'")
        return text