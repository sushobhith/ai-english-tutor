import tempfile
import unittest
from pathlib import Path

from version import get_package_version


class VersionTest(unittest.TestCase):
    def test_reads_version_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            version_file = Path(temp_dir) / "VERSION"
            version_file.write_text("2.3.1\n", encoding="utf-8")

            self.assertEqual(get_package_version(version_file), "2.3.1")

    def test_rejects_empty_version_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            version_file = Path(temp_dir) / "VERSION"
            version_file.write_text("\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                get_package_version(version_file)


if __name__ == "__main__":
    unittest.main()
