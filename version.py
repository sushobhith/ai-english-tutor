from pathlib import Path

VERSION_FILE = Path(__file__).with_name("VERSION")


def get_package_version() -> str:
    version = VERSION_FILE.read_text(encoding="utf-8").strip()
    if not version:
        raise ValueError("VERSION file is empty")
    return version


async def version_command(update, context):
    await update.message.reply_text(f"AI English Tutor version {get_package_version()}")
