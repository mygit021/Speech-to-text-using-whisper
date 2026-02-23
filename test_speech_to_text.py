"""Tests for speech_to_text_app utility functions."""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from speech_to_text_app import (
    get_default_output_path,
    save_text_to_file,
    transcribe_audio,
    SAMPLE_RATE,
)


class TestGetDefaultOutputPath(unittest.TestCase):
    """Tests for get_default_output_path."""

    def test_returns_txt_file_in_cwd(self):
        path = get_default_output_path()
        self.assertTrue(path.endswith(".txt"))
        self.assertEqual(os.path.dirname(path), os.getcwd())


class TestSaveTextToFile(unittest.TestCase):
    """Tests for save_text_to_file."""

    def test_creates_file_and_writes_text(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            save_text_to_file("hello world", tmp_path)
            with open(tmp_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("hello world", content)
            # Should contain a timestamp in brackets
            self.assertIn("[", content)
            self.assertIn("]", content)
        finally:
            os.unlink(tmp_path)

    def test_appends_multiple_lines(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            save_text_to_file("first line", tmp_path)
            save_text_to_file("second line", tmp_path)
            with open(tmp_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("first line", lines[0])
            self.assertIn("second line", lines[1])
        finally:
            os.unlink(tmp_path)


class TestTranscribeAudio(unittest.TestCase):
    """Tests for transcribe_audio with mocked Whisper model."""

    def test_returns_transcribed_text(self):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": " Hello, how are you? "}
        audio = np.zeros(SAMPLE_RATE * 3, dtype=np.float32)

        result = transcribe_audio(mock_model, audio)

        self.assertEqual(result, "Hello, how are you?")
        mock_model.transcribe.assert_called_once()

    def test_returns_empty_string_for_empty_result(self):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": ""}
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)

        result = transcribe_audio(mock_model, audio)

        self.assertEqual(result, "")

    def test_handles_missing_text_key(self):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {}
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)

        result = transcribe_audio(mock_model, audio)

        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
