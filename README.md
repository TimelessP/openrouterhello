# Openrouterhello

Openrouterhello is a Python CLI demo for multimodal chat with OpenRouter using the OpenAI SDK.

## Features

- Text input prompts
- Microphone recording (WAV)
- Attach audio files from URL or path (`.wav`, `.mp3`)
- Attach image from URL or path (paths are converted to data URLs)
- Attach MP4 video from URL or path (paths are converted to data URLs)
- Local tool calling support with two tools:
  - `get_current_utc_datetime`
  - `generate_uuid4`
- Live tool call logging during execution

## Requirements

- Python 3.10+
- OpenRouter API key
- Optional audio recording dependencies:
  - `sounddevice`
  - `soundfile`

## Setup

1. Install dependencies:

   ```bash
   .venv/bin/pip install -r requirements.txt
   ```

2. Configure environment variables in `.env`:

   ```env
   OPENROUTER_API_KEY=your_key_here
   OPENROUTER_MODEL=openrouter/healer-alpha
   ```

   `OPENROUTER_MODEL` is optional. If not set, the default model is `openrouter/healer-alpha`.

## Run

```bash
python main.py
```

## Menu

- `1` Input text
- `2` Record audio
- `3` Attach WAV/MP3 file (URL or file path)
- `4` Attach image (URL or file path)
- `5` Attach MP4 video (URL or file path)
- `6` Send
- `9` Exit

## Notes

- Some multimodal/audio requests may require paid balance depending on provider/model.
- Supported audio formats can vary by provider.
