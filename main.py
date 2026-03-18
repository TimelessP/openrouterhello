import os
import sys
import base64
import tempfile
from typing import Any, cast

from dotenv import load_dotenv
from openai import OpenAI


def _audio_dependencies_available() -> bool:
    try:
        import sounddevice  # noqa: F401
        import soundfile  # noqa: F401
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


AUDIO_AVAILABLE = _audio_dependencies_available()

load_dotenv()

SAMPLE_RATE = 44100
CHANNELS = 1
MODEL = "openrouter/healer-alpha"


def record_audio() -> str | None:
    """Record audio until the user presses Enter. Returns base64-encoded WAV or None."""
    try:
        import sounddevice as sd
        import soundfile as sf
        import numpy as np
    except ImportError:
        print("Audio dependencies missing. Install sounddevice and soundfile first.")
        return None

    frames = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    print("Recording... press Enter to stop.")
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                            dtype="int16", callback=callback):
            input()
    except Exception as e:
        print(f"Recording error: {e}")
        return None

    if not frames:
        print("No audio captured.")
        return None

    audio_data = np.concatenate(frames, axis=0)

    tmp_path = tempfile.mktemp(suffix=".wav")
    try:
        sf.write(tmp_path, audio_data, SAMPLE_RATE)
        with open(tmp_path, "rb") as f:
            wav_bytes = f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    duration = len(audio_data) / SAMPLE_RATE
    print(f"Recorded {duration:.1f}s of audio.")
    return base64.b64encode(wav_bytes).decode("utf-8")


def send(client: OpenAI, pending_text: str | None, pending_audio: str | None):
    content: list[dict[str, Any]] = []
    if pending_text:
        content.append({"type": "text", "text": pending_text})
    if pending_audio:
        content.append({
            "type": "input_audio",
            "input_audio": {"data": pending_audio, "format": "wav"},
        })

    # Use simple string for text-only (broader model compatibility)
    if pending_text and not pending_audio:
        messages = [{"role": "user", "content": pending_text}]
    else:
        messages = [{"role": "user", "content": content}]

    print("\nSending to OpenRouter...")
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://localhost",
                "X-OpenRouter-Title": "CLI Demo",
            },
            extra_body={},
            model=MODEL,
            messages=cast(Any, messages),
        )
        print("\n--- Response ---")
        print(completion.choices[0].message.content)
        print("----------------")
    except Exception as e:
        print(f"\nAPI error: {e}")


def main():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found. Check your .env file.")
        sys.exit(1)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    pending_text: str | None = None
    pending_audio: str | None = None

    print("OpenRouter CLI Demo")
    print(f"Model: {MODEL}")
    print("=" * 40)

    while True:
        print()
        if pending_text:
            preview = pending_text[:60] + ("..." if len(pending_text) > 60 else "")
            print(f"  [text]  {preview}")
        if pending_audio:
            print("  [audio] ready")
        print()
        print("  1. Input text")
        if AUDIO_AVAILABLE:
            print("  2. Record audio")
        else:
            print("  2. Record audio  (unavailable — pip install sounddevice soundfile)")
        print("  3. Send")
        print("  9. Exit")

        choice = input("\n> ").strip()

        if choice == "1":
            text = input("Enter your message: ").strip()
            if text:
                pending_text = text
                print("Text stored.")
            else:
                print("No text entered.")

        elif choice == "2":
            if not AUDIO_AVAILABLE:
                print("Install sounddevice and soundfile first.")
                continue
            audio = record_audio()
            if audio:
                pending_audio = audio

        elif choice == "3":
            if not pending_text and not pending_audio:
                print("Nothing to send. Add text or record audio first.")
                continue
            send(client, pending_text, pending_audio)
            pending_text = None
            pending_audio = None

        elif choice == "9":
            print("Goodbye!")
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
