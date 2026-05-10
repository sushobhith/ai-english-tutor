import asyncio
import unittest
from pathlib import Path

from version import VERSION_FILE, get_package_version, version_command


class FakeMessage:
    def __init__(self):
        self.replies = []

    async def reply_text(self, text):
        self.replies.append(text)


class FakeUpdate:
    def __init__(self):
        self.message = FakeMessage()


class VersionCommandTest(unittest.TestCase):
    def test_version_file_contains_semantic_version(self):
        version = get_package_version()

        self.assertRegex(version, r"^\d+\.\d+\.\d+$")
        self.assertEqual(version, VERSION_FILE.read_text(encoding="utf-8").strip())

    def test_version_command_replies_with_current_version(self):
        update = FakeUpdate()

        asyncio.run(version_command(update, None))

        self.assertEqual(len(update.message.replies), 1)
        self.assertIn(get_package_version(), update.message.replies[0])

    def test_main_registers_version_command(self):
        main_source = Path("main.py").read_text(encoding="utf-8")

        self.assertIn("from version import version_command", main_source)
        self.assertRegex(
            main_source,
            r"CommandHandler\(\s*['\"]version['\"]\s*,\s*version_command\s*\)",
        )


if __name__ == "__main__":
    unittest.main()
