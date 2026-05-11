from pathlib import Path


VERSION_FILE = Path(__file__).with_name("VERSION")


def get_package_version() -> str:
    return VERSION_FILE.read_text(encoding="utf-8").strip()
