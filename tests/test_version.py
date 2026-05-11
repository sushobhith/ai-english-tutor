import unittest

from version import get_package_version


class VersionTestCase(unittest.TestCase):
    def test_get_package_version_returns_version_file_value(self):
        self.assertRegex(get_package_version(), r"^\d+\.\d+\.\d+$")


if __name__ == "__main__":
    unittest.main()
