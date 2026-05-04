import asyncio
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


def install_runtime_stubs():
    analyse = types.ModuleType("analyse")
    analyse.get_completion = lambda _text: "{}"
    sys.modules["analyse"] = analyse

    dotenv = types.ModuleType("dotenv")
    dotenv.find_dotenv = lambda: ""
    dotenv.load_dotenv = lambda _path=None: None
    sys.modules["dotenv"] = dotenv

    speech_recognition = types.ModuleType("speech_recognition")
    speech_recognition.Recognizer = object
    speech_recognition.UnknownValueError = Exception
    speech_recognition.RequestError = Exception
    speech_recognition.AudioFile = object
    sys.modules["speech_recognition"] = speech_recognition

    reportlab = types.ModuleType("reportlab")
    reportlab_lib = types.ModuleType("reportlab.lib")
    reportlab_pagesizes = types.ModuleType("reportlab.lib.pagesizes")
    reportlab_pagesizes.letter = (612, 792)
    reportlab_pdfgen = types.ModuleType("reportlab.pdfgen")
    reportlab_canvas = types.ModuleType("reportlab.pdfgen.canvas")
    reportlab_canvas.Canvas = object
    sys.modules["reportlab"] = reportlab
    sys.modules["reportlab.lib"] = reportlab_lib
    sys.modules["reportlab.lib.pagesizes"] = reportlab_pagesizes
    sys.modules["reportlab.pdfgen"] = reportlab_pdfgen
    sys.modules["reportlab.pdfgen.canvas"] = reportlab_canvas

    telegram = types.ModuleType("telegram")
    telegram.Update = object
    sys.modules["telegram"] = telegram

    telegram_ext = types.ModuleType("telegram.ext")

    class StubApplication:
        last_instance = None

        def __init__(self):
            self.handlers = []
            self.error_handlers = []
            StubApplication.last_instance = self

        @classmethod
        def builder(cls):
            return StubApplicationBuilder()

        def add_handler(self, handler):
            self.handlers.append(handler)

        def add_error_handler(self, handler):
            self.error_handlers.append(handler)

        def run_polling(self, poll_interval):
            self.poll_interval = poll_interval

    class StubApplicationBuilder:
        def token(self, token):
            self.token_value = token
            return self

        def build(self):
            return StubApplication()

    class StubCommandHandler:
        def __init__(self, command, callback):
            self.command = command
            self.callback = callback

    class StubMessageHandler:
        def __init__(self, message_filter, callback):
            self.message_filter = message_filter
            self.callback = callback

    class StubFilters:
        TEXT = "text"
        VOICE = "voice"

    class StubContextTypes:
        DEFAULT_TYPE = object

    telegram_ext.Application = StubApplication
    telegram_ext.CommandHandler = StubCommandHandler
    telegram_ext.MessageHandler = StubMessageHandler
    telegram_ext.filters = StubFilters
    telegram_ext.ContextTypes = StubContextTypes
    sys.modules["telegram.ext"] = telegram_ext


install_runtime_stubs()
import main


class VersionCommandTest(unittest.TestCase):
    def test_version_file_contains_semantic_version(self):
        version = main.get_version()

        self.assertRegex(version, r"^\d+\.\d+\.\d+$")

    def test_get_version_reads_supplied_version_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            version_file = Path(temp_dir) / "VERSION"
            version_file.write_text("2.3.1\n", encoding="utf-8")

            self.assertEqual(main.get_version(version_file), "2.3.1")

    def test_version_command_replies_with_version(self):
        update = AsyncMock()

        with patch("main.get_version", return_value="2.3.1"):
            asyncio.run(main.version_command(update, AsyncMock()))

        update.message.reply_text.assert_awaited_once_with("Version: 2.3.1")

    def test_main_registers_version_command(self):
        with patch("builtins.print"):
            main.main()

        command_handlers = [
            handler.command
            for handler in main.Application.last_instance.handlers
            if isinstance(handler, main.CommandHandler)
        ]
        self.assertIn("version", command_handlers)


if __name__ == "__main__":
    unittest.main()
