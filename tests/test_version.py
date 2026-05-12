from pathlib import Path
import unittest

from version import VERSION_FILE, get_package_version


class VersionCommandTests(unittest.TestCase):
    def test_version_file_contains_semver(self):
        version = get_package_version()

        self.assertTrue(version)
        self.assertEqual(version.count("."), 2)
        self.assertTrue(all(part.isdigit() for part in version.split(".")))

    def test_version_helper_reads_version_file(self):
        self.assertEqual(
            get_package_version(),
            VERSION_FILE.read_text(encoding="utf-8").strip(),
        )

    def test_version_command_is_registered(self):
        main_source = Path("main.py").read_text(encoding="utf-8")

        self.assertIn("async def version_command", main_source)
        self.assertIn("CommandHandler('version', version_command)", main_source)


if __name__ == "__main__":
    unittest.main()
