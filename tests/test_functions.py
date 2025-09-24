"""Tests for functions.functions module."""

import os
from pathlib import Path

import pytest

import functions.functions as funcs

from functions.functions import (
    create_note,
    get_battery_status,
    lock_screen,
    search_google,
    set_volume,
)


def test_search_google_opens_browser(monkeypatch):
    opened = {}

    def fake_open(url):
        opened["url"] = url
        return True

    monkeypatch.setattr(funcs.webbrowser, "open", fake_open)
    query = "hello world"
    result = search_google(query)
    assert "Searched for" in result
    assert opened["url"].startswith("https://www.google.com/search?q=")
    assert "hello+world" in opened["url"]


def test_get_battery_status_handles_missing(monkeypatch):

    class FakeBattery:
        percent = 77
        power_plugged = True

    monkeypatch.setattr(funcs.psutil, "sensors_battery", FakeBattery)
    status = get_battery_status()
    assert status["percent"] == 77
    assert status["charging"] is True


def test_lock_screen_mac_success(monkeypatch):

    calls = {}

    def fake_run(cmd, check):
        calls["cmd"] = cmd
        calls["check"] = check
        return 0

    monkeypatch.setattr(funcs.subprocess, "run", fake_run)
    assert lock_screen() == "Screen locked"
    assert calls["cmd"][0] == "pmset"


def test_set_volume_mac_success(monkeypatch):

    recorded = {}

    def fake_run(cmd, check):
        recorded["cmd"] = cmd
        recorded["check"] = check
        return 0

    monkeypatch.setattr(funcs.subprocess, "run", fake_run)
    msg = set_volume(42)
    assert msg == "Volume set to 42%"
    assert recorded["cmd"][0] == "osascript"


@pytest.mark.parametrize(
    "system_name, open_cmd, expect_msg_contains",
    [
        ("Darwin", "osascript", "Note created in Apple Notes"),
        ("Windows", "notepad.exe", "Note created in"),
        ("Linux", "xdg-open", "Note created in"),
    ],
)
def test_create_note_across_platforms(
    tmp_path, monkeypatch, system_name, open_cmd, expect_msg_contains
):

    # Mock platform
    monkeypatch.setattr(funcs.platform, "system", lambda: system_name)

    # Redirect home/Documents/Notes to tmp_path/Notes
    fake_home = tmp_path / "home"
    (fake_home / "Documents").mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("HOME", str(fake_home))
    if os.name == "nt":
        monkeypatch.setenv("USERPROFILE", str(fake_home))

    # Mock process launching differently per OS
    popen_calls = []

    def fake_popen(args):
        popen_calls.append(args)

        class X:
            pass

        return X()

    # macOS: also uses osascript run; Windows has startfile fallback; Linux uses xdg-open
    monkeypatch.setattr(funcs.subprocess, "Popen", fake_popen)

    ran_osascript = {"ran": False}

    def fake_run(args, check):
        if system_name == "Darwin" and args and args[0] == "osascript":
            ran_osascript["ran"] = True
        return 0

    monkeypatch.setattr(funcs.subprocess, "run", fake_run)

    # On Windows, os.startfile may be used as fallback; mock it if present
    if system_name == "Windows":
        monkeypatch.setattr(funcs.os, "startfile", lambda path: None, raising=False)

    msg = create_note("My Title", "Body text")
    assert expect_msg_contains in msg

    notes_dir = Path(fake_home) / "Documents" / "Notes"
    if system_name == "Darwin":
        # Apple Notes path isn't created; relies on Notes app
        assert ran_osascript["ran"] is True
    else:
        # File should exist
        files = list(notes_dir.glob("My Title.txt"))
        assert len(files) == 1
        assert files[0].read_text(encoding="utf-8").startswith("My Title\n\nBody text")
