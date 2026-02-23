"""Speech-to-Text application using OpenAI Whisper tiny model.

A graphical application that captures audio from the microphone in real time,
transcribes it using the Whisper tiny model, and displays the text on screen
while also saving it to a text file.
"""

import os
import threading
import datetime
import queue
import tkinter as tk
from tkinter import scrolledtext, filedialog

import numpy as np
import sounddevice as sd
import whisper

# Audio recording parameters
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
CHANNELS = 1
DTYPE = "float32"
RECORD_SECONDS = 5  # Duration of each recording chunk in seconds
POLL_INTERVAL_MS = 200  # Interval in ms for polling the transcription queue


def get_default_output_path():
    """Return a default output file path for the transcription."""
    return os.path.join(os.getcwd(), "transcription_output.txt")


def load_whisper_model(model_name="tiny"):
    """Load and return the Whisper model.

    Args:
        model_name: Name of the Whisper model to load. Defaults to 'tiny'.

    Returns:
        The loaded Whisper model.
    """
    return whisper.load_model(model_name)


def record_audio_chunk(duration, sample_rate=SAMPLE_RATE, channels=CHANNELS):
    """Record a chunk of audio from the microphone.

    Args:
        duration: Duration in seconds to record.
        sample_rate: Audio sample rate in Hz.
        channels: Number of audio channels.

    Returns:
        Numpy array of recorded audio samples.
    """
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype=DTYPE,
    )
    sd.wait()
    return audio.flatten()


def transcribe_audio(model, audio_data, sample_rate=SAMPLE_RATE):
    """Transcribe audio data using the Whisper model.

    Args:
        model: Loaded Whisper model.
        audio_data: Numpy array of audio samples (float32).
        sample_rate: Sample rate of the audio data.

    Returns:
        Transcribed text string.
    """
    # Whisper expects float32 audio at 16kHz
    audio_data = audio_data.astype(np.float32)
    result = model.transcribe(audio_data, fp16=False)
    return result.get("text", "").strip()


def save_text_to_file(text, filepath):
    """Append transcribed text to a file.

    Args:
        text: Text to save.
        filepath: Path to the output file.
    """
    with open(filepath, "a", encoding="utf-8") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {text}\n")


class SpeechToTextApp:
    """Main GUI application for speech-to-text conversion."""

    def __init__(self, root):
        self.root = root
        self.root.title("Speech to Text - Whisper")
        self.root.geometry("700x550")
        self.root.minsize(500, 400)

        self.model = None
        self.is_recording = False
        self.output_filepath = get_default_output_path()
        self.result_queue = queue.Queue()

        self._build_ui()
        self._poll_results()

    def _build_ui(self):
        """Build the application UI."""
        # Title label
        title_label = tk.Label(
            self.root,
            text="Speech to Text (Whisper Tiny)",
            font=("Helvetica", 16, "bold"),
        )
        title_label.pack(pady=(10, 5))

        # Status label
        self.status_var = tk.StringVar(value="Status: Ready – Click 'Start Recording'")
        status_label = tk.Label(
            self.root, textvariable=self.status_var, font=("Helvetica", 10)
        )
        status_label.pack(pady=(0, 5))

        # Button frame
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)

        self.start_btn = tk.Button(
            btn_frame,
            text="Start Recording",
            command=self._start_recording,
            width=18,
            bg="#4CAF50",
            fg="white",
            font=("Helvetica", 11),
        )
        self.start_btn.grid(row=0, column=0, padx=5)

        self.stop_btn = tk.Button(
            btn_frame,
            text="Stop Recording",
            command=self._stop_recording,
            width=18,
            bg="#f44336",
            fg="white",
            font=("Helvetica", 11),
            state=tk.DISABLED,
        )
        self.stop_btn.grid(row=0, column=1, padx=5)

        self.save_btn = tk.Button(
            btn_frame,
            text="Choose Output File",
            command=self._choose_output_file,
            width=18,
            font=("Helvetica", 11),
        )
        self.save_btn.grid(row=0, column=2, padx=5)

        # Output file path display
        self.filepath_var = tk.StringVar(value=f"Output: {self.output_filepath}")
        filepath_label = tk.Label(
            self.root, textvariable=self.filepath_var, font=("Helvetica", 9),
            wraplength=680, fg="gray",
        )
        filepath_label.pack(pady=(5, 5))

        # Transcription text area
        text_label = tk.Label(
            self.root, text="Transcription:", font=("Helvetica", 11, "bold"),
            anchor="w",
        )
        text_label.pack(fill=tk.X, padx=10)

        self.text_area = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, font=("Helvetica", 11), state=tk.DISABLED
        )
        self.text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 5))

        # Clear button
        clear_btn = tk.Button(
            self.root, text="Clear Text", command=self._clear_text,
            font=("Helvetica", 10),
        )
        clear_btn.pack(pady=(0, 10))

    def _poll_results(self):
        """Poll the result queue and update the UI with transcribed text."""
        try:
            while True:
                text = self.result_queue.get_nowait()
                if text:
                    self._append_text(text)
                    save_text_to_file(text, self.output_filepath)
        except queue.Empty:
            pass
        self.root.after(POLL_INTERVAL_MS, self._poll_results)

    def _start_recording(self):
        """Start the recording loop in a background thread."""
        self.is_recording = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("Status: Loading Whisper model...")

        thread = threading.Thread(target=self._recording_loop, daemon=True)
        thread.start()

    def _stop_recording(self):
        """Stop the recording loop."""
        self.is_recording = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Status: Stopped")

    def _recording_loop(self):
        """Background loop that records and transcribes audio chunks."""
        # Load model on first use
        if self.model is None:
            try:
                self.model = load_whisper_model("tiny")
            except Exception as e:
                self.result_queue.put(None)
                self.root.after(0, lambda: self.status_var.set(
                    f"Error loading model: {e}"
                ))
                self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
                self.is_recording = False
                return

        self.root.after(0, lambda: self.status_var.set("Status: Recording..."))

        while self.is_recording:
            try:
                audio_data = record_audio_chunk(RECORD_SECONDS)
                if not self.is_recording:
                    break
                self.root.after(
                    0, lambda: self.status_var.set("Status: Transcribing...")
                )
                text = transcribe_audio(self.model, audio_data)
                if text:
                    self.result_queue.put(text)
                if self.is_recording:
                    self.root.after(
                        0, lambda: self.status_var.set("Status: Recording...")
                    )
            except Exception as e:
                self.root.after(
                    0, lambda err=e: self.status_var.set(f"Error: {err}")
                )
                break

    def _append_text(self, text):
        """Append text to the scrolled text area."""
        self.text_area.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, text + "\n")
        self.text_area.see(tk.END)
        self.text_area.config(state=tk.DISABLED)

    def _clear_text(self):
        """Clear the text area."""
        self.text_area.config(state=tk.NORMAL)
        self.text_area.delete("1.0", tk.END)
        self.text_area.config(state=tk.DISABLED)

    def _choose_output_file(self):
        """Open a file dialog to choose the output text file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Choose output file",
        )
        if filepath:
            self.output_filepath = filepath
            self.filepath_var.set(f"Output: {self.output_filepath}")


def main():
    """Entry point for the application."""
    root = tk.Tk()
    SpeechToTextApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
