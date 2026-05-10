from pathlib import Path


VERSION_FILE = Path(__file__).with_name("VERSION")


def get_package_version(version_file: Path = VERSION_FILE) -> str:
    version = version_file.read_text(encoding="utf-8").strip()
    if not version:
        raise ValueError(f"{version_file} is empty")
    return version
