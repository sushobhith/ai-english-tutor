from pathlib import Path


VERSION_FILE = Path(__file__).with_name("VERSION")


def get_package_version() -> str:
    """Return the project version from the root VERSION file."""
    return VERSION_FILE.read_text(encoding="utf-8").strip()
