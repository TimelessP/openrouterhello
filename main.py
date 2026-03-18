import os
import sys
import base64
import json
import mimetypes
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
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
MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/healer-alpha")


def get_current_utc_datetime() -> dict[str, str]:
    now = datetime.now(timezone.utc)
    return {
        "utc_datetime": now.isoformat(),
        "unix_timestamp": str(int(now.timestamp())),
    }


def generate_uuid4() -> dict[str, str]:
    return {"uuid4": str(uuid.uuid4())}


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_utc_datetime",
            "description": "Get the current date and time in UTC.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_uuid4",
            "description": "Generate a fresh UUIDv4 string.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]


TOOL_FUNCTIONS: dict[str, Any] = {
    "get_current_utc_datetime": get_current_utc_datetime,
    "generate_uuid4": generate_uuid4,
}


def _is_web_or_data_url(value: str) -> bool:
    return value.startswith(("http://", "https://", "data:"))


def _to_data_url(file_path: str, default_mime: str, force_mime: str | None = None) -> str | None:
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.exists() or not path.is_file():
        print(f"File not found: {file_path}")
        return None

    mime_type, _ = mimetypes.guess_type(str(path))
    if force_mime:
        mime_type = force_mime
    elif not mime_type:
        mime_type = default_mime

    try:
        raw = path.read_bytes()
    except OSError as e:
        print(f"Could not read file: {e}")
        return None

    size_mb = len(raw) / (1024 * 1024)
    print(f"Loaded {path.name} ({size_mb:.2f} MB).")
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def resolve_image_input(value: str) -> str | None:
    if _is_web_or_data_url(value):
        return value

    data_url = _to_data_url(value, default_mime="image/jpeg")
    if not data_url:
        return None
    if not data_url.startswith("data:image/"):
        print("Warning: file may not be an image based on mime type.")
    return data_url


def resolve_mp4_input(value: str) -> str | None:
    if _is_web_or_data_url(value):
        return value

    if Path(value).suffix.lower() != ".mp4":
        print("Video file path must point to an .mp4 file.")
        return None

    return _to_data_url(value, default_mime="video/mp4", force_mime="video/mp4")


def _media_preview(value: str) -> str:
    if value.startswith("data:"):
        header = value.split(",", 1)[0]
        return f"{header},..."
    return value if len(value) <= 60 else value[:57] + "..."


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


def _execute_tool_call(tool_call: Any) -> dict[str, str]:
    tool_name = tool_call.function.name
    raw_args = tool_call.function.arguments or "{}"

    try:
        parsed_args = json.loads(raw_args)
        if not isinstance(parsed_args, dict):
            parsed_args = {}
    except json.JSONDecodeError:
        parsed_args = {}

    tool_fn = TOOL_FUNCTIONS.get(tool_name)
    if tool_fn is None:
        result: dict[str, str] = {"error": f"Unknown tool: {tool_name}"}
    else:
        try:
            result = tool_fn(**parsed_args)
        except TypeError as e:
            result = {"error": f"Invalid args for {tool_name}: {e}"}
        except Exception as e:
            result = {"error": f"Tool failed ({tool_name}): {e}"}

    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_name,
        "content": json.dumps(result),
    }


def _respond_with_tools(client: OpenAI, messages: list[dict[str, Any]]) -> str:
    max_rounds = 4

    for _ in range(max_rounds):
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://localhost",
                "X-OpenRouter-Title": "CLI Demo",
            },
            extra_body={},
            model=MODEL,
            messages=cast(Any, messages),
            tools=cast(Any, TOOLS),
            tool_choice="auto",
            parallel_tool_calls=True,
        )

        message = completion.choices[0].message
        message_dict = cast(dict[str, Any], message.model_dump(exclude_none=True))
        messages.append(message_dict)

        tool_calls = message.tool_calls
        if not tool_calls:
            return message.content or ""

        print("\n--- Tool calls ---")
        for tool_call in tool_calls:
            tool_call_any = cast(Any, tool_call)
            raw_args = tool_call_any.function.arguments or "{}"
            print(f"[tool/request] {tool_call_any.function.name} args={raw_args}")

        tool_results: list[dict[str, str]] = [
            {"role": "tool", "tool_call_id": "", "name": "", "content": ""}
            for _ in tool_calls
        ]

        with ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
            future_to_index = {
                executor.submit(_execute_tool_call, tool_call): idx
                for idx, tool_call in enumerate(tool_calls)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                tool_result = future.result()
                tool_results[idx] = tool_result
                print(f"[tool/result] {tool_result['name']} -> {tool_result['content']}")

        print("------------------")

        messages.extend(tool_results)

    return "I reached the tool-call round limit before producing a final answer."


def send(
    client: OpenAI,
    pending_text: str | None,
    pending_audio: str | None,
    pending_image: str | None,
    pending_video: str | None,
):
    content: list[dict[str, Any]] = []
    if pending_text:
        content.append({"type": "text", "text": pending_text})
    if pending_image:
        content.append({
            "type": "image_url",
            "image_url": {"url": pending_image},
        })
    if pending_audio:
        content.append({
            "type": "input_audio",
            "input_audio": {"data": pending_audio, "format": "wav"},
        })
    if pending_video:
        content.append({
            "type": "video_url",
            "video_url": {"url": pending_video},
        })

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You can call tools when useful. "
                "Use get_current_utc_datetime for current UTC date/time and "
                "generate_uuid4 for a fresh UUID."
            ),
        }
    ]

    # Use simple string for text-only (broader model compatibility)
    if pending_text and not pending_audio and not pending_image and not pending_video:
        messages.append({"role": "user", "content": pending_text})
    else:
        messages.append({"role": "user", "content": content})

    print("\nSending to OpenRouter...")
    try:
        final_text = _respond_with_tools(client, messages)
        print("\n--- Response ---")
        print(final_text)
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
    pending_image: str | None = None
    pending_video: str | None = None

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
        if pending_image:
            print(f"  [image] {_media_preview(pending_image)}")
        if pending_video:
            print(f"  [video] {_media_preview(pending_video)}")
        print()
        print("  1. Input text")
        if AUDIO_AVAILABLE:
            print("  2. Record audio")
        else:
            print("  2. Record audio  (unavailable — pip install sounddevice soundfile)")
        print("  4. Attach image (URL or file path)")
        print("  5. Attach MP4 video (URL or file path)")
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
            if not pending_text and not pending_audio and not pending_image and not pending_video:
                print("Nothing to send. Add text, audio, image, or video first.")
                continue
            send(client, pending_text, pending_audio, pending_image, pending_video)
            pending_text = None
            pending_audio = None
            pending_image = None
            pending_video = None

        elif choice == "4":
            image_input = input("Image URL or file path: ").strip()
            if not image_input:
                print("No image input.")
                continue
            image_value = resolve_image_input(image_input)
            if image_value:
                pending_image = image_value
                print("Image attached.")

        elif choice == "5":
            video_input = input("MP4 URL or file path: ").strip()
            if not video_input:
                print("No video input.")
                continue
            video_value = resolve_mp4_input(video_input)
            if video_value:
                pending_video = video_value
                print("Video attached.")

        elif choice == "9":
            print("Goodbye!")
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
