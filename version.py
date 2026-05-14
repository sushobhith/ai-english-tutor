from importlib import metadata
from pathlib import Path


PACKAGE_NAME = "ai-english-tutor"
VERSION_FILE = Path(__file__).with_name("VERSION")


def get_package_version() -> str:
    """Return the installed package version or the source tree version."""
    try:
        return metadata.version(PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return VERSION_FILE.read_text(encoding="utf-8").strip()
