"Tool Functions"

import subprocess
import webbrowser
import os
import platform
from pathlib import Path

import psutil


def lock_screen() -> str:
    try:
        subprocess.run(["pmset", "displaysleepnow"], check=True)
        return "Screen locked"
    except (subprocess.CalledProcessError, FileNotFoundError, PermissionError) as e:
        return f"Failed to lock screen: {e}"


def get_battery_status() -> dict:
    battery = psutil.sensors_battery()
    if battery is None:
        return {"error": "Battery information not available"}
    try:
        return {"percent": battery.percent, "charging": battery.power_plugged}
    except (AttributeError,) as e:
        return {"error": str(e)}


def search_google(query: str) -> str:
    try:
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        if not webbrowser.open(url):
            return "Failed to open browser"
        return f"Searched for: {query}"
    except (webbrowser.Error, OSError) as e:
        return f"Failed to search: {e}"


def set_volume(level: int) -> str:
    try:
        subprocess.run(
            ["osascript", "-e", f"set volume output volume {level}"], check=True
        )
        return f"Volume set to {level}%"
    except (subprocess.CalledProcessError, FileNotFoundError, PermissionError) as e:
        return f"Failed to set volume: {e}"


def create_note(title: str, content: str) -> str:
    try:
        current_os = platform.system()

        if current_os == "Darwin":
            # macOS: Use Apple Notes via AppleScript
            subprocess.Popen(["open", "-a", "Notes"])  # best-effort open
            applescript = f"""
            tell application "Notes"
                tell account "iCloud"
                    make new note with properties {{name:"{title}", body:"{content}"}}
                end tell
                activate
            end tell
            """
            subprocess.run(["osascript", "-e", applescript], check=True)
            return f"Note created in Apple Notes: {title}"

        # Windows/Linux fallback: create a text file and open with default editor
        def sanitize_filename(name: str) -> str:
            # Remove characters illegal in Windows filenames and trim whitespace
            illegal_chars = '<>:"/\\|?*'
            sanitized = "".join(c for c in name if c not in illegal_chars)
            sanitized = sanitized.strip().rstrip(".")  # avoid trailing dot
            return sanitized or "untitled"

        notes_dir = Path.home() / "Documents" / "Notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{sanitize_filename(title)}.txt"
        file_path = notes_dir / filename
        file_path.write_text(f"{title}\n\n{content}", encoding="utf-8")

        if current_os == "Windows":
            # Prefer opening with Notepad
            try:
                subprocess.Popen(["notepad.exe", str(file_path)])
            except OSError:
                os.startfile(str(file_path))  # type: ignore[attr-defined]
            return f"Note created in {file_path} and opened with Notepad"

        if current_os == "Linux":
            # Open with default editor via xdg-open
            subprocess.Popen(["xdg-open", str(file_path)])
            return f"Note created in {file_path} and opened with default editor"

        # Other/unknown OS: try to open generically
        try:
            if hasattr(os, "startfile"):
                os.startfile(str(file_path))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["open", str(file_path)])
        except (OSError, FileNotFoundError):
            pass
        return f"Note created in {file_path}"
    except (OSError, FileNotFoundError, subprocess.CalledProcessError) as e:
        return f"Failed to create note: {e}"


available_function_calls = {
    "lock_screen": lock_screen,
    "get_battery_status": get_battery_status,
    "search_google": search_google,
    "set_volume": set_volume,
    "create_note": create_note,
}
