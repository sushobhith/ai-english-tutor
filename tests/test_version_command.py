import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def install_stub_modules():
    dotenv = types.ModuleType('dotenv')
    dotenv.find_dotenv = lambda: ''
    dotenv.load_dotenv = lambda *_args, **_kwargs: None
    sys.modules['dotenv'] = dotenv

    telegram = types.ModuleType('telegram')
    telegram.Update = object
    sys.modules['telegram'] = telegram

    telegram_ext = types.ModuleType('telegram.ext')
    telegram_ext.ContextTypes = types.SimpleNamespace(DEFAULT_TYPE=object)
    telegram_ext.Application = object
    telegram_ext.CommandHandler = lambda *args, **kwargs: ('command', args, kwargs)
    telegram_ext.MessageHandler = lambda *args, **kwargs: ('message', args, kwargs)
    telegram_ext.filters = types.SimpleNamespace(TEXT=object(), VOICE=object())
    sys.modules['telegram.ext'] = telegram_ext

    reportlab = types.ModuleType('reportlab')
    reportlab_lib = types.ModuleType('reportlab.lib')
    reportlab_pagesizes = types.ModuleType('reportlab.lib.pagesizes')
    reportlab_pagesizes.letter = (612, 792)
    reportlab_pdfgen = types.ModuleType('reportlab.pdfgen')
    reportlab_canvas = types.ModuleType('reportlab.pdfgen.canvas')
    reportlab_canvas.Canvas = object
    sys.modules['reportlab'] = reportlab
    sys.modules['reportlab.lib'] = reportlab_lib
    sys.modules['reportlab.lib.pagesizes'] = reportlab_pagesizes
    sys.modules['reportlab.pdfgen'] = reportlab_pdfgen
    sys.modules['reportlab.pdfgen.canvas'] = reportlab_canvas

    sys.modules['requests'] = types.ModuleType('requests')
    sys.modules['speech_recognition'] = types.ModuleType('speech_recognition')
    sys.modules['analyse'] = types.ModuleType('analyse')


def load_main():
    install_stub_modules()
    spec = importlib.util.spec_from_file_location('main', ROOT / 'main.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyMessage:
    def __init__(self):
        self.replies = []

    async def reply_text(self, text):
        self.replies.append(text)


class VersionCommandTest(unittest.TestCase):
    def test_version_file_contains_semver(self):
        version = (ROOT / 'VERSION').read_text(encoding='utf-8').strip()

        self.assertRegex(version, r'^\d+\.\d+\.\d+$')

    def test_get_package_version_reads_version_file(self):
        main = load_main()

        self.assertEqual(main.get_package_version(), '1.0.0')

    def test_version_command_replies_with_version(self):
        main = load_main()
        message = DummyMessage()
        update = types.SimpleNamespace(message=message)

        asyncio.run(main.version_command(update, object()))

        self.assertEqual(message.replies, ['AI English Tutor version 1.0.0'])


if __name__ == '__main__':
    unittest.main()
