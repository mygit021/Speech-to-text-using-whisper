# Speech-to-Text Using Whisper

A Python graphical application that converts speech to text in real time using
[OpenAI Whisper](https://github.com/openai/whisper) (tiny model). The app
captures audio from your microphone, transcribes it on screen, and saves the
output to a text file.

## Features

- **Real-time recording** – captures audio from your microphone in chunks
- **Whisper tiny model** – fast, lightweight transcription via OpenAI Whisper
- **Live display** – transcribed text appears in a scrollable text area
- **File output** – every transcription is appended (with timestamps) to a text file
- **Choose output file** – pick any `.txt` file to save your transcriptions

## Requirements

- Python 3.9+
- A working microphone
- [PortAudio](http://www.portaudio.com/) (required by `sounddevice`)
  - **Ubuntu/Debian:** `sudo apt-get install libportaudio2`
  - **macOS:** `brew install portaudio`
  - **Windows:** included with the `sounddevice` pip package

## Installation

```bash
# Clone the repository
git clone https://github.com/mygit021/Speech-to-text-using-whisper.git
cd Speech-to-text-using-whisper

# Install Python dependencies
pip install -r requirements.txt
```

## Usage

```bash
python speech_to_text_app.py
```

1. Click **Start Recording** to begin capturing audio (the Whisper model loads
   on first use).
2. Speak into your microphone – transcribed text appears in the text area.
3. Click **Stop Recording** to pause.
4. Use **Choose Output File** to change where transcriptions are saved
   (default: `transcription_output.txt` in the current directory).
5. Use **Clear Text** to clear the on-screen transcription.

## Running Tests

```bash
python -m unittest test_speech_to_text -v
```

## License

See [LICENSE](LICENSE) for details.